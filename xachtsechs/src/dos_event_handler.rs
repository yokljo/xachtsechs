use crate::dos_file_system::DosFileSystem;
use crate::types::{EventHandler, Flag, InterruptResult, Reg, RegHalf};
use crate::machine8086::{INTERRUPT_TABLE_ENTRY_BYTES, Machine8086};
use crate::bios_loader::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MachineType {
	EGA,
}

impl MachineType {
	pub fn lookup_video_mode(&self, mode_index: u8) -> Result<VideoMode, String> {
		match self {
			MachineType::EGA => {
				for video_mode in &EGA_MODES {
					if video_mode.mode_index == mode_index {
						return Ok(video_mode.clone());
					}
				}
			}
		}
		
		Err(format!("Couldn't find video mode for {:?}: 0x{:02x}", self, mode_index))
	}
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum VGAMode {
	Text,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoMode {
	mode_index: u8,
	vga_mode: VGAMode,
	pixel_dims: (u32, u32),
	// Number of columns/rows of text on the screen.
	text_dims: (u32, u32),
	// Size of each character in pixels.
	char_pixel_dims: (u32, u32),
	// This is where the text data starts in memory. Each character consists of an ASCII byte for
	// the character, and a byte representing the colour.
	text_address: u32,
	// This is the number of "pages" of text available in this video mode.
	text_page_count: u32,
	// This is the number of bytes per page in memory.
	text_page_bytes: u32,
}

pub const EGA_MODES: [VideoMode; 1] = [
	VideoMode {
		mode_index: 3, vga_mode: VGAMode::Text, pixel_dims: (640, 480), text_dims: (80, 25),
		char_pixel_dims: (8, 14), text_address: 0xb8000, text_page_count: 8, text_page_bytes: 0x1000,
	},
];

#[derive(Debug, Clone, PartialEq)]
pub struct PortStates {
	port_61: u16,
	crt_index_register: u16,
	cga_status_register: u16,
}

impl PortStates {
	pub fn new() -> PortStates {
		PortStates {
			port_61: 0,
			crt_index_register: 0,
			cga_status_register: 0,
		}
	}
}

#[derive(Debug)]
pub struct DosEventHandler {
	pub machine_type: MachineType,
	pub video_mode: VideoMode,
	pub port_states: PortStates,
	pub file_system: Box<DosFileSystem>,
	pub seconds_since_start: f64,
}

impl DosEventHandler {
	pub fn init_machine(&mut self, machine: &mut Machine8086) {
		//self.set_video_mode(3);
		machine.set_data_u8(&BIOS_VIDEO_MODE_INDEX, self.video_mode.mode_index);
		machine.set_data_u16(&BIOS_TEXT_COLUMN_COUNT, self.video_mode.text_dims.0 as u16);
		machine.set_data_u16(&BIOS_TEXT_PAGE_BYTES, self.video_mode.text_page_bytes as u16);
		machine.set_data_u16(&BIOS_VIDEO_IO_PORT_ADDRESS, 0x3d4 as u16);
		machine.set_data_u16(&BIOS_TEXT_ROW_COUNT, self.video_mode.text_dims.1 as u16);
		machine.set_data_u16(&BIOS_CHAR_HEIGHT, self.video_mode.char_pixel_dims.1 as u16);
	}

	/*fn set_video_mode(&mut self, machine: &mut Machine8086, mode_index: u8) {
		self.video_mode = self.lookup_video_mode(mode_index).unwrap();
		
	}*/
	
	pub fn set_cga_vertial_retrace(&mut self, vertical_retrace: bool) {
		if vertical_retrace {
			self.port_states.cga_status_register |= 0b1000u16;
		} else {
			self.port_states.cga_status_register &= !0b1000u16;
		}
	}

	fn get_page_origin_address(&self, machine: &Machine8086, video_page: u8) -> u32 {
		let page_bytes = machine.get_data_u16(&BIOS_TEXT_PAGE_BYTES);
		self.video_mode.text_address + (video_page as u32 * page_bytes as u32)
	}
	
	fn get_screen_character_address(&self, machine: &Machine8086, page_origin_address: u32, x: u8, y: u8) -> u32 {
		let bytes_per_char = 2;
		let column_count = machine.get_data_u16(&BIOS_TEXT_COLUMN_COUNT);
		page_origin_address + (((y as u32 * column_count as u32) + x as u32) * bytes_per_char)
	}
	
	fn handle_interrupt_10h(&mut self, machine: &mut Machine8086) {
		// Video (http://www.ctyme.com/intr/int-10.htm)
		let video_int = machine.get_reg_u8(Reg::AX, RegHalf::High);
		println!("Video interrupt: 0x{:x}", video_int);
		match video_int {
			0x02 => {
				// Set cursor position.
				let bh = machine.get_reg_u8(Reg::BX, RegHalf::High);
				let video_page = if bh == 0xff { machine.get_data_u8(&BIOS_ACTIVE_VIDEO_PAGE) } else { bh };
				
				let dh = machine.get_reg_u8(Reg::DX, RegHalf::High);
				let dl = machine.get_reg_u8(Reg::DX, RegHalf::High);
				let cursor_pos_data = ((dh as u16) << 8) + dl as u16;
				machine.set_data_u16(&BIOS_CURSOR_POSITION[video_page as usize], cursor_pos_data);
			}
			0x06 => {
				// Scroll the text up within a rectangular area on the active page.
				let video_page = machine.get_data_u8(&BIOS_ACTIVE_VIDEO_PAGE);
				let num_lines = machine.get_reg_u8(Reg::AX, RegHalf::Low);
				let blank_char_attributes = machine.get_reg_u8(Reg::BX, RegHalf::High);
				let rect_top = machine.get_reg_u8(Reg::CX, RegHalf::High);
				let rect_left = machine.get_reg_u8(Reg::CX, RegHalf::Low);
				let rect_bottom = machine.get_reg_u8(Reg::DX, RegHalf::High);
				let rect_right = machine.get_reg_u8(Reg::DX, RegHalf::Low);
				let page_addr = self.get_page_origin_address(machine, video_page);
				
				if num_lines == 0 {
					// Clear the window.
					for y in rect_top ..= rect_bottom {
						for x in rect_left ..= rect_right {
							let char_addr = self.get_screen_character_address(machine, page_addr, x, y);
							machine.poke_u8(char_addr, 0);
							machine.poke_u8(char_addr + 1, blank_char_attributes);
						}
					}
				} else {
					for y in rect_top ..= (rect_bottom - num_lines) {
						for x in rect_left ..= rect_right {
							let from_addr = self.get_screen_character_address(machine, page_addr, x, y + 1);
							let to_addr = self.get_screen_character_address(machine, page_addr, x, y);
							let char_data = machine.peek_u16(from_addr);
							machine.poke_u16(to_addr, char_data);
						}
					}
					for y in (rect_bottom - num_lines + 1) ..= rect_bottom {
						for x in rect_left ..= rect_right {
							let char_addr = self.get_screen_character_address(machine, page_addr, x, y);
							machine.poke_u8(char_addr, 0);
							machine.poke_u8(char_addr + 1, blank_char_attributes);
						}
					}
				}
			}
			0x08 => {
				// Read char and attributes at cursor position
				// TODO
				/*let bh = machine.get_reg_u8(Reg::BX, RegHalf::High);
				let bl = machine.get_reg_u8(Reg::BX, RegHalf::Low);
				let video_page = if bh == 0xff { machine.get_data_u8(&BIOS_ACTIVE_VIDEO_PAGE) } else { bh };
				let (cursor_x, cursor_y) = split_u16_high_low(machine.get_data_u16(&BIOS_CURSOR_POSITION[video_page as usize]));
				let page_bytes = machine.get_data_u16(&BIOS_TEXT_PAGE_BYTES);
				let column_count = machine.get_data_u16(&BIOS_TEXT_COLUMN_COUNT);
				let bytes_per_char = 2;
				let addr = self.video_mode.text_address + (video_page as u32 * page_bytes as u32) + (((cursor_y as u32 * column_count as u32) + cursor_x as u32) * bytes_per_char);
				dbg!((addr, video_page, column_count, cursor_x, cursor_y, bl, bh));
				let char_colour_attrs = machine.peek_u16(addr);
				machine.set_reg_u16(Reg::AX, char_colour_attrs);*/
				
				// Copying ZETA to test that it actually works:
				let bh = machine.get_reg_u8(Reg::BX, RegHalf::High);
				let bl = machine.get_reg_u8(Reg::BX, RegHalf::Low);
				let addr = self.video_mode.text_address + (((bl as u32 * 80 as u32) + bh as u32) * 2);
				machine.set_reg_u8(Reg::AX, RegHalf::Low, machine.peek_u8(addr));
				machine.set_reg_u8(Reg::BX, RegHalf::High, machine.peek_u8(addr + 1));
			}
			0x0f => {
				// Get current video mode
				let text_column_count = machine.get_data_u16(&BIOS_TEXT_COLUMN_COUNT);
				machine.set_reg_u8(Reg::AX, RegHalf::High, text_column_count as u8);
				// Video modes covered in: http://www.ctyme.com/intr/rb-0069.htm
				// 3 is the 80x25 colour mode
				machine.set_reg_u8(Reg::AX, RegHalf::Low, 3);
				// Active display page (http://www.ctyme.com/intr/rb-0091.htm)
				machine.set_reg_u8(Reg::BX, RegHalf::High, machine.get_data_u8(&BIOS_ACTIVE_VIDEO_PAGE));
			}
			0x11 => {
				let func11 = machine.get_reg_u8(Reg::AX, RegHalf::Low);
				match func11 {
					0x30 => {
						// TODO: Get font information
						
						// Copying ZETA:
						machine.set_flag(Flag::Carry, true);
					}
					_ => panic!("Unknown video 0x11 func: 0x{:x}", func11)
				}
			}
			_ => panic!("Unknown video func: 0x{:x}", video_int)
		}
	}
}

impl EventHandler for DosEventHandler {
	fn handle_interrupt(&mut self, machine: &mut Machine8086, interrupt_index: u8) -> InterruptResult {
		// https://www.shsu.edu/~csc_tjm/spring2001/cs272/interrupt.html
		println!("Handle interrupt: 0x{:x}", interrupt_index);
		match interrupt_index {
			// BIOS Interrupts (0x00-0x1F):
			0x02 => {
				// Non-maskable interrupt
				panic!("Memory corruption error, apparently...");
			}
			0x04 => {
				// Overflow
				panic!("Overflow");
			}
			0x08 => {
				// Timer interrupt. This is supposed to be injected by an external source exactly
				// 18.2 times per second.
				// TODO
				let timer_low = machine.get_data_u16(&BIOS_SYSTEM_TIMER_COUNTER_LOW);
				let timer_high = machine.get_data_u16(&BIOS_SYSTEM_TIMER_COUNTER_HIGH);
				let timer = timer_low as u32 + ((timer_high as u32) << 16);
				let new_timer = timer.wrapping_add(1);
				println!("Time: {}", new_timer);
				let new_timer_low = (new_timer & 0xffff) as u16;
				let new_timer_high = ((new_timer >> 16) & 0xffff) as u16;
				machine.set_data_u16(&BIOS_SYSTEM_TIMER_COUNTER_LOW, new_timer_low);
				machine.set_data_u16(&BIOS_SYSTEM_TIMER_COUNTER_HIGH, new_timer_high);
				// Emit user timer tick.
				machine.interrupt_on_next_step(0x1c);
			}
			0x10 => {
				self.handle_interrupt_10h(machine);
			}
			0x14 => {
				// Serial port services
				let serial_int = machine.get_reg_u8(Reg::AX, RegHalf::High);
				println!("Serial port interrupt: {}", serial_int);
			}
			0x1c => {
				// User timer tick, emitted by 0x08.
			}
			
			// This is the DOS interrupt.
			// http://spike.scu.edu.au/~barry/interrupts.html
			// http://stanislavs.org/helppc/int_21.html
			0x21 => {
				let dos_int = machine.get_reg_u8(Reg::AX, RegHalf::High);
				println!("DOS Interrupt: 0x{:x}", dos_int);
				match dos_int {
					0x25 => {
						// Get ES:BX and store it as an entry of the interrupt vector/table (as the IP:CS).
						let entry_addr = machine.get_reg_u8(Reg::AX, RegHalf::Low) as u32 * INTERRUPT_TABLE_ENTRY_BYTES as u32;
						let interrupt_ip = machine.get_reg_u16(Reg::DX);
						let interrupt_cs = machine.get_reg_u16(Reg::DS);
						machine.poke_u16(entry_addr, interrupt_ip);
						machine.poke_u16(entry_addr + 2, interrupt_cs);
					}
					0x2c => {
						// Get system time.
						let hundredths = ((self.seconds_since_start * 100.) as usize % 100) as u8;
						let second = (self.seconds_since_start as usize % 60) as u8;
						let minute = ((self.seconds_since_start / 60.) as usize % 60) as u8;
						let hour = ((self.seconds_since_start / 60. / 60.) as usize % 24) as u8;
						machine.set_reg_u8(Reg::CX, RegHalf::High, hour);
						machine.set_reg_u8(Reg::CX, RegHalf::Low, minute);
						machine.set_reg_u8(Reg::DX, RegHalf::High, second);
						machine.set_reg_u8(Reg::DX, RegHalf::Low, hundredths);
					}
					0x33 => {
						// Modify Ctrl+Break shortcut functionality.
						// TODO
						machine.set_reg_u8(Reg::DX, RegHalf::Low, 0);
					}
					0x35 => {
						// Get an entry of the interrupt vector/table (IP:CS) and store it in ES:BX.
						let entry_addr = machine.get_reg_u8(Reg::AX, RegHalf::Low) as u32 * INTERRUPT_TABLE_ENTRY_BYTES as u32;
						let interrupt_ip = machine.peek_u16(entry_addr);
						let interrupt_cs = machine.peek_u16(entry_addr + 2);
						machine.set_reg_u16(Reg::BX, interrupt_ip);
						machine.set_reg_u16(Reg::ES, interrupt_cs);
					}
					0x3c => {
						// CREAT
						let filename_addr = machine.get_seg_reg(Reg::DS, Reg::DX);
						let filename = machine.read_null_terminated_string(filename_addr);
						let attributes = machine.get_reg_u16(Reg::CX);
						match self.file_system.create(filename, attributes) {
							Ok(handle) => {
								machine.set_flag(Flag::Carry, false);
								machine.set_reg_u16(Reg::AX, handle);
							}
							Err(error_code) => {
								machine.set_flag(Flag::Carry, true);
								machine.set_reg_u16(Reg::AX, error_code);
							}
						}
					}
					0x3d => {
						// OPEN
						let filename_addr = machine.get_seg_reg(Reg::DS, Reg::DX);
						let filename = machine.read_null_terminated_string(filename_addr);
						let access_modes = machine.get_reg_u8(Reg::AX, RegHalf::Low);
						match self.file_system.open(filename, access_modes) {
							Ok(handle) => {
								machine.set_flag(Flag::Carry, false);
								machine.set_reg_u16(Reg::AX, handle);
							}
							Err(error_code) => {
								machine.set_flag(Flag::Carry, true);
								machine.set_reg_u16(Reg::AX, error_code);
							}
						}
					}
					0x44 => {
						// I/O control
						let io_func = machine.get_reg_u8(Reg::AX, RegHalf::Low);
						match io_func {
							0 => {
								// Get device information
								// TODO
								machine.set_reg_u16(Reg::AX, 1);
								machine.set_flag(Flag::Carry, true);
							}
							_ => println!("Unknown IO func: 0x{:x}", io_func)
						}
					}
					_ => panic!("Unknown DOS interrupt: 0x{:x}", dos_int)
				}
			}
			0x33 => {
				// Mouse function calls
				// http://stanislavs.org/helppc/int_33.html
				let mouse_func = machine.get_reg_u16(Reg::AX);
				match mouse_func {
					0 => {
						// TODO get mouse installed flag
					}
					_ => panic!("Unknown mouse function: 0x{:x}", mouse_func)
				}
			}
			_ => panic!("Unknown interrupt: 0x{:x}", interrupt_index)
		}
		
		InterruptResult::Return
	}
	
	fn handle_port_input(&mut self, machine: &mut Machine8086, port_index: u16) -> u16 {
		// http://bochs.sourceforge.net/techspec/PORTS.LST
		match port_index {
			0x61 => {
				// "Keyboard Controller" control register.
				// TODO
				self.port_states.port_61
			}
			0x201 => {
				// TODO: Read joystick values.
				0
			}
			0x3da => {
				let status = self.port_states.cga_status_register;
				self.set_cga_vertial_retrace(false);
				status
			}
			_ => panic!("Unhandled input port index: 0x{:02x}", port_index)
		}
	}
	
	fn handle_port_output(&mut self, machine: &mut Machine8086, port_index: u16, value: u16) {
		match port_index {
			0x61 => {
				// TODO
				self.port_states.port_61 = value;
			}
			0x201 => {
				// TODO: Something about joystick one-shots?
			}
			0x3d4 => {
				self.port_states.crt_index_register = value;
			}
			0x3d5 => {
				// TODO: CRT data register
			}
			_ => panic!("Unhandled output port index: 0x{:02x}", port_index)
		}
	}
}
