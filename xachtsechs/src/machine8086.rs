use crate::types::{Address16, AdjustMode, AnyUnsignedInt8086, ArithmeticMode, DataLocation, DataLocation8, DataLocation16, DisplacementOrigin, EventHandler, Flag, Inst, InterruptResult, JumpCondition, JumpConditionType, ModRmRegMode, OpDirection, OpSize, Reg, REG_COUNT, RegHalf, ScalarMode, ShiftMode, SourceDestination, treat_i8_as_u8, treat_u8_as_i8, treat_i16_as_u16, treat_u16_as_i16, treat_i32_as_u32, treat_u32_as_i32, u8_on_bits_are_odd};

use num::FromPrimitive;
use std::collections::VecDeque;

/// This is the size of one entry in the interrupt table. Each entry consists of a IP, followed by
/// the associated CS.
pub const INTERRUPT_TABLE_ENTRY_BYTES: usize = 2 + 2;

pub struct Machine8086 {
	memory: Vec<u8>,
	registers: [u16; REG_COUNT],
	halted: bool,
	pending_interrupts: VecDeque<u8>,
	
	override_default_segment: Option<Reg>,
	number_of_parsed_instructions: usize,
}

impl Machine8086 {
	pub fn new(memory_bytes: usize) -> Machine8086 {
		let mut machine = Machine8086 {
			memory: vec![0; memory_bytes],
			registers: [0; REG_COUNT as usize],
			halted: false,
			pending_interrupts: VecDeque::new(),
			
			override_default_segment: None,
			number_of_parsed_instructions: 0,
		};
		
		machine.set_flag(Flag::Interrupt, true);
		// This bit is set by default in ZETA for some reason.
		machine.registers[Reg::Flags as usize] |= 0b10;
		
		// Configure the interrupt table.
		for i in 0..256u16 {
			let addr = i as u32 * INTERRUPT_TABLE_ENTRY_BYTES as u32;
			// The IP to jump to.
			machine.poke_u16(addr, 0x1100 + i);
			// The CS the IP is within.
			machine.poke_u16(addr + 2, 0xf000);
		}
		
		machine
	}
	
	pub fn insert_contiguous_bytes(&mut self, bytes: &[u8], at: usize) {
		self.memory.splice(at..(at + bytes.len()), bytes.iter().cloned());
	}
	
	pub fn calculate_displacement(&self, origin: DisplacementOrigin) -> u16 {
		match origin {
			DisplacementOrigin::BX_SI => self.get_reg_u16(Reg::BX).wrapping_add(self.get_reg_u16(Reg::SI)),
			DisplacementOrigin::BX_DI => self.get_reg_u16(Reg::BX).wrapping_add(self.get_reg_u16(Reg::DI)),
			DisplacementOrigin::BP_SI => self.get_reg_u16(Reg::BP).wrapping_add(self.get_reg_u16(Reg::SI)),
			DisplacementOrigin::BP_DI => self.get_reg_u16(Reg::BP).wrapping_add(self.get_reg_u16(Reg::DI)),
			DisplacementOrigin::SI => self.get_reg_u16(Reg::SI),
			DisplacementOrigin::DI => self.get_reg_u16(Reg::DI),
			DisplacementOrigin::BP => self.get_reg_u16(Reg::BP),
			DisplacementOrigin::BX => self.get_reg_u16(Reg::BX),
		}
	}
	
	pub fn calculate_effective_address16(&self, address16: &Address16) -> u16 {
		match *address16 {
			Address16::Reg(reg) => {
				self.get_reg_u16(reg)
			}
			Address16::DisplacementRel(displacement, offset) => {
				self.calculate_displacement(displacement).wrapping_add(offset)
			}
			Address16::Immediate(addr) => addr,
		}
	}
	
	pub fn get_data_u8(&self, location: &DataLocation8) -> u8 {
		match *location {
			DataLocation8::Reg(reg, half) => {
				self.get_reg_u8(reg, half)
			}
			DataLocation8::MemoryAbs(addr) => self.peek_u8(addr),
			DataLocation8::Memory{seg, ref address16} => {
				let addr = self.get_seg_offset(seg, self.calculate_effective_address16(&address16));
				self.peek_u8(addr)
			}
			DataLocation8::Immediate(value) => value,
		}
	}
	
	pub fn set_data_u8(&mut self, location: &DataLocation8, value: u8) {
		match *location {
			DataLocation8::Reg(reg, half) => {
				self.set_reg_u8(reg, half, value);
			}
			DataLocation8::MemoryAbs(addr) => self.poke_u8(addr, value),
			DataLocation8::Memory{seg, ref address16} => {
				let addr = self.get_seg_offset(seg, self.calculate_effective_address16(&address16));
				self.poke_u8(addr, value);
			}
			DataLocation8::Immediate(value) => panic!("Attempted to use immediate as destination: {}", value),
		}
	}

	pub fn add_to_data_u8(&mut self, location: &DataLocation8, amount: u8) {
		let value = self.get_data_u8(location);
		self.set_data_u8(location, value.wrapping_add(amount));
	}

	pub fn sub_from_data_u8(&mut self, location: &DataLocation8, amount: u8) {
		let value = self.get_data_u8(location);
		self.set_data_u8(location, value.wrapping_sub(amount));
	}
	
	pub fn get_data_u16(&self, location: &DataLocation16) -> u16 {
		match *location {
			DataLocation16::Reg(reg) => {
				self.get_reg_u16(reg)
			}
			DataLocation16::MemoryAbs(addr) => self.peek_u16(addr),
			DataLocation16::Memory{seg, ref address16} => {
				let addr = self.get_seg_offset(seg, self.calculate_effective_address16(&address16));
				self.peek_u16(addr)
			}
			DataLocation16::Address16(ref address16) => {
				self.calculate_effective_address16(&address16)
			}
			DataLocation16::Immediate(value) => value,
		}
	}
	
	pub fn set_data_u16(&mut self, location: &DataLocation16, value: u16) {
		match *location {
			DataLocation16::Reg(reg) => {
				self.set_reg_u16(reg, value);
			}
			DataLocation16::MemoryAbs(addr) => self.poke_u16(addr, value),
			DataLocation16::Memory{seg, ref address16} => {
				let addr = self.get_seg_offset(seg, self.calculate_effective_address16(&address16));
				self.poke_u16(addr, value);
			}
			DataLocation16::Address16(ref address16) => panic!("Attempted to use address as destination: {}", value),
			DataLocation16::Immediate(value) => panic!("Attempted to use immediate as destination: {}", value),
		}
	}
	
	pub fn add_to_data_u16(&mut self, location: &DataLocation16, amount: u16) {
		let value = self.get_data_u16(location);
		self.set_data_u16(location, value.wrapping_add(amount));
	}
	
	pub fn sub_from_data_u16(&mut self, location: &DataLocation16, amount: u16) {
		let value = self.get_data_u16(location);
		self.set_data_u16(location, value.wrapping_sub(amount));
	}
	
	pub fn get_reg_u8(&self, reg: Reg, half: RegHalf) -> u8 {
		let value16 = self.registers[reg as usize];
		match half {
			RegHalf::High => (value16 >> 8) as u8,
			RegHalf::Low => (value16 & 0xff) as u8,
		}
	}
	
	pub fn set_reg_u8(&mut self, reg: Reg, half: RegHalf, value: u8) {
		let value16 = &mut self.registers[reg as usize];
		match half {
			RegHalf::High => {
				*value16 = (*value16 & 0x00ff) | ((value as u16) << 8);
			}
			RegHalf::Low => {
				*value16 = (*value16 & 0xff00) | (value as u16);
			}
		}
	}

	pub fn get_reg_u16(&self, reg: Reg) -> u16 {
		self.registers[reg as usize]
	}
	
	pub fn set_reg_u16(&mut self, reg: Reg, value: u16) {
		self.registers[reg as usize] = value;
	}
	
	pub fn add_to_reg(&mut self, reg: Reg, amount: u16) {
		let reg_value = &mut self.registers[reg as usize];
		*reg_value = reg_value.wrapping_add(amount);
	}
	
	pub fn sub_from_reg(&mut self, reg: Reg, amount: u16) {
		let reg_value = &mut self.registers[reg as usize];
		*reg_value = reg_value.wrapping_sub(amount);
	}

	// 8086 has 20 address bits
	// http://www.renyujie.net/articles/article_ca_x86_4.php
	pub fn get_seg_offset(&self, seg_reg: Reg, offset: u16) -> u32 {
		(((self.get_reg_u16(seg_reg) as u32) << 4) + offset as u32) & 0xfffff
	}

	pub fn get_seg_reg(&self, seg_reg: Reg, reg: Reg) -> u32 {
		self.get_seg_offset(seg_reg, self.get_reg_u16(reg))
	}
	
	pub fn get_ip(&self) -> u32 {
		self.get_seg_reg(Reg::CS, Reg::IP)
	}
	
	pub fn get_sp(&self) -> u32 {
		self.get_seg_reg(Reg::SS, Reg::SP)
	}
	
	pub fn peek_u8(&self, at: u32) -> u8 {
		self.memory[at as usize]
	}
	
	pub fn poke_u8(&mut self, at: u32, value: u8) {
		self.memory[at as usize] = value;
	}
	
	pub fn peek_u16(&self, at: u32) -> u16 {
		((self.memory[at as usize + 1] as u16) << 8) + self.memory[at as usize] as u16
	}
	
	pub fn poke_u16(&mut self, at: u32, value: u16) {
		if at >= 4098 && at < 4100 {
			println!("poke_u8 {} {}", at, value);
		}
	
		self.memory[at as usize] = (value & 0x00ff) as u8;
		self.memory[at as usize + 1] = ((value & 0xff00) >> 8) as u8;
	}
	
	pub fn read_null_terminated_string(&self, start: u32) -> Vec<u8> {
		let mut result = vec![];
		let mut current = start;
		loop {
			let c = self.peek_u8(current);
			if c == 0 {
				break;
			}
			result.push(c);
			current += 1;
		}
		result
	}
	
	pub fn push_u16(&mut self, value: u16) {
		//println!("Push16({})", value);
		self.sub_from_reg(Reg::SP, 2);
		self.poke_u16(self.get_sp(), value);
	}
	
	pub fn pop_u16(&mut self) -> u16 {
		let value = self.peek_u16(self.get_sp());
		//println!("Pop16({})", value);
		self.add_to_reg(Reg::SP, 2);
		value
	}
	
	pub fn read_ip_u8(&mut self) -> u8 {
		let value = self.peek_u8(self.get_ip());
		self.add_to_reg(Reg::IP, 1);
		value
	}
	
	pub fn read_ip_u16(&mut self) -> u16 {
		let value = self.peek_u16(self.get_ip());
		self.add_to_reg(Reg::IP, 2);
		value
	}
	
	pub fn set_flag(&mut self, flag: Flag, on: bool) {
		let mut flags = self.get_reg_u16(Reg::Flags);
		let bit = 0b1 << (flag as u16);
		if on {
			flags = flags | bit;
		} else {
			flags = flags & (!bit);
		}
		self.set_reg_u16(Reg::Flags, flags);
	}
	
	pub fn get_flag(&self, flag: Flag) -> bool {
		let flags = self.get_reg_u16(Reg::Flags);
		let bit = 0b1 << (flag as u16);
		(flags & bit) != 0
	}
	
	pub fn test_jump_condition(&self, condition: &JumpCondition) -> bool {
		let pos_result = match condition.condition {
			JumpConditionType::Overflow => self.get_flag(Flag::Overflow),
			JumpConditionType::Carry => self.get_flag(Flag::Carry),
			JumpConditionType::Zero => self.get_flag(Flag::Zero),
			JumpConditionType::CarryOrZero => self.get_flag(Flag::Carry) || self.get_flag(Flag::Zero),
			JumpConditionType::Sign => self.get_flag(Flag::Sign),
			JumpConditionType::Parity => self.get_flag(Flag::Parity),
			JumpConditionType::OverlowIsNotSign => self.get_flag(Flag::Overflow) != self.get_flag(Flag::Sign),
			JumpConditionType::OverlowIsNotSignOrZero => self.get_flag(Flag::Overflow) != self.get_flag(Flag::Sign) || self.get_flag(Flag::Zero),
		};
		if condition.negate { !pos_result } else { pos_result }
	}
	
	fn resolve_default_segment(&self, default_segment: Reg) -> Reg {
		if let Some(override_default_segment) = self.override_default_segment {
			override_default_segment
		} else {
			default_segment
		}
	}
	
	fn read_modrm_source_destination(&mut self, opsize: OpSize, opdir: OpDirection, reg_mode: ModRmRegMode) -> SourceDestination {
		let modrm_code = self.read_ip_u8();
		//println!("Modrm code: {:08b}", modrm_code);
		let rm_code = modrm_code & 0b00000111;
		let reg_code = (modrm_code & 0b00111000) >> 3;
		let mod_code = (modrm_code & 0b11000000) >> 6;
		
		let mut first: DataLocation = match reg_mode {
			ModRmRegMode::Reg => match opsize {
				OpSize::Size8 => {
					let (reg, half) = Reg::reg8(reg_code as usize).unwrap();
					DataLocation::Size8(DataLocation8::Reg(reg, half))
				}
				OpSize::Size16 => {
					let reg = Reg::reg16(reg_code as usize).unwrap();
					DataLocation::Size16(DataLocation16::Reg(reg))
				}
			}
			ModRmRegMode::Seg => match opsize {
				OpSize::Size8 => panic!("Reg cannot be seg in 8 bit mode"),
				OpSize::Size16 => DataLocation::Size16(DataLocation16::Reg(Reg::seg_reg(reg_code as usize).unwrap())),
			}
			ModRmRegMode::Imm => match opsize {
				OpSize::Size8 => DataLocation::Size8(DataLocation8::Immediate(reg_code)),
				OpSize::Size16 => DataLocation::Size16(DataLocation16::Immediate(reg_code as u16)),
			}
		};
		
		let mut second: Option<DataLocation> = None;
		let mut displacement: Option<u16> = None;
		
		let displacement_origin = match rm_code {
			0b000 => DisplacementOrigin::BX_SI,
			0b001 => DisplacementOrigin::BX_DI,
			0b010 => DisplacementOrigin::BP_SI,
			0b011 => DisplacementOrigin::BP_DI,
			0b100 => DisplacementOrigin::SI,
			0b101 => DisplacementOrigin::DI,
			0b110 => DisplacementOrigin::BP,
			0b111 => DisplacementOrigin::BX,
			_ => unreachable!()
		};
		
		match mod_code {
			0b00 => {
				// If r/m is 110, Displacement (16 bits) is address; otherwise, no displacement
				if rm_code == 0b110 {
					let addr = self.read_ip_u16();
					// Displacements are relative to the data segment (DS)
					let seg = self.resolve_default_segment(Reg::DS);
					second = Some(match opsize {
						OpSize::Size8 => DataLocation::Size8(DataLocation8::Memory{seg, address16: Address16::Immediate(addr)}),
						OpSize::Size16 => DataLocation::Size16(DataLocation16::Memory{seg, address16: Address16::Immediate(addr)}),
					});
				} else {
					displacement = Some(0);
				};
				
			}
			0b01 => {
				// Eight-bit displacement, sign-extended to 16 bits
				displacement = Some(treat_i16_as_u16(treat_u8_as_i8(self.read_ip_u8()) as i16));
			}
			0b10 => {
				// 16-bit displacement (example: MOV [BX + SI]+ displacement,al)
				displacement = Some(self.read_ip_u16());
			}
			0b11 => {
				// r/m is treated as a second "reg" field
				second = Some(match opsize {
					OpSize::Size8 => {
						let (reg, half) = Reg::reg8(rm_code as usize).unwrap();
						DataLocation::Size8(DataLocation8::Reg(reg, half))
					}
					OpSize::Size16 => DataLocation::Size16(DataLocation16::Reg(Reg::reg16(rm_code as usize).unwrap()))
				});
			}
			_ => unreachable!()
		}
		
		if let Some(offset) = displacement {
			let seg = self.resolve_default_segment(displacement_origin.default_segment());
			second = Some(match opsize {
				OpSize::Size8 => DataLocation::Size8(DataLocation8::Memory{seg, address16: Address16::DisplacementRel(displacement_origin, offset)}),
				OpSize::Size16 => DataLocation::Size16(DataLocation16::Memory{seg, address16: Address16::DisplacementRel(displacement_origin, offset)}),
			});
		}
		
		match (first, second) {
			(DataLocation::Size8(source), Some(DataLocation::Size8(destination))) => {
				match opdir {
					OpDirection::Source => SourceDestination::Size8(source, destination),
					OpDirection::Destination => SourceDestination::Size8(destination, source),
				}
			}
			(DataLocation::Size16(source), Some(DataLocation::Size16(destination))) => {
				match opdir {
					OpDirection::Source => SourceDestination::Size16(source, destination),
					OpDirection::Destination => SourceDestination::Size16(destination, source),
				}
			}
			_ => panic!("ModR/M byte didn't fully resolve")
		}
	}
	
	fn read_modrm_with_immediate_reg_u8(&mut self) -> (u8, DataLocation8) {
		let source_destination = self.read_modrm_source_destination(OpSize::Size8, OpDirection::Source, ModRmRegMode::Imm);
		let (imm, destination) = match source_destination {
			SourceDestination::Size8(source, destination) => match source {
				DataLocation8::Immediate(imm) => (imm, destination),
				_ => panic!("Source was not an immediate value"),
			}
			SourceDestination::Size16(_, _) => panic!("Source was not 8 bits"),
		};
		(imm, destination)
	}
	
	fn read_modrm_with_immediate_reg_u16(&mut self) -> (u16, DataLocation16) {
		let source_destination = self.read_modrm_source_destination(OpSize::Size16, OpDirection::Source, ModRmRegMode::Imm);
		let (imm, destination) = match source_destination {
			SourceDestination::Size8(_, _) => panic!("Source was not 16 bits"),
			SourceDestination::Size16(source, destination) => match source {
				DataLocation16::Immediate(imm) => (imm, destination),
				_ => panic!("Source was not an immediate value"),
			}
		};
		(imm, destination)
	}
	
	fn read_standard_source_destination(&mut self, opcode: u8, reg_mode: ModRmRegMode) -> SourceDestination {
		let opsize = if opcode & 0b01 == 0 { OpSize::Size8 } else { OpSize::Size16 };
		let opdir = if opcode & 0b10 == 0 { OpDirection::Source } else { OpDirection::Destination };
		self.read_modrm_source_destination(opsize, opdir, reg_mode)
	}
	
	fn read_arithmetic_source_destination_with_ax(&mut self, opcode: u8) -> SourceDestination {
		match opcode & 0b111 {
			0x00 ... 0x03 => self.read_standard_source_destination(opcode, ModRmRegMode::Reg),
			0x04 => {
				let imm = self.read_ip_u8();
				SourceDestination::Size8(DataLocation8::Immediate(imm), DataLocation8::Reg(Reg::AX, RegHalf::Low))
			}
			0x05 => {
				let imm = self.read_ip_u16();
				SourceDestination::Size16(DataLocation16::Immediate(imm), DataLocation16::Reg(Reg::AX))
			}
			_ => panic!("Attempted to call read_arithmetic_source_destination with bad opcode: 0x{:x}", opcode)
		}
	}
	
	fn get_standard_scalar_mode(&self, index: u8) -> ScalarMode {
		match index {
			4 => ScalarMode::Mul,
			5 => ScalarMode::SignedMul,
			6 => ScalarMode::Div,
			7 => ScalarMode::SignedDiv,
			_ => panic!("Non-standard scalar mode: {}", index)
		}
	}

	pub fn parse_instruction(&mut self) -> Inst {
		let opcode = self.read_ip_u8();
		
		if self.number_of_parsed_instructions >= 697000 {
			//println!("{:?}", self.registers);
			//println!("Opcode: 0x{:02x} ({:?})", opcode, self.number_of_parsed_instructions);
		}
		self.number_of_parsed_instructions += 1;
		/*if self.number_of_parsed_instructions > 697516 {//697762 {
			panic!();
		}*/
		//println!("IP: {:?}", self.get_ip());
		match opcode {
			0x00 ... 0x05 => Inst::Arithmetic(ArithmeticMode::Add, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x06 => Inst::Push16(DataLocation16::Reg(Reg::ES)),
			0x07 => Inst::Pop16(DataLocation16::Reg(Reg::ES)),
			0x08 ... 0x0d => Inst::Arithmetic(ArithmeticMode::Or, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x0e => Inst::Push16(DataLocation16::Reg(Reg::CS)),
			0x0f => {
				// This opcode is undocumented. Apparently it's a fluke that it exists, because the
				// CPU doesn't consider all the bits when processing an instruction, so it ends up
				// as a duplicate of a different instruction (probably 0x1f).
				// See: http://www.os2museum.com/wp/undocumented-8086-opcodes-part-i/
				Inst::Pop16(DataLocation16::Reg(Reg::CS))
			}
			0x10 ... 0x15 => Inst::Arithmetic(ArithmeticMode::AddWithCarry, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x16 => Inst::Push16(DataLocation16::Reg(Reg::SS)),
			0x17 => Inst::Pop16(DataLocation16::Reg(Reg::SS)),
			0x18 ... 0x1d => Inst::Arithmetic(ArithmeticMode::SubWithBorrow, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x1e => Inst::Push16(DataLocation16::Reg(Reg::DS)),
			0x1f => Inst::Pop16(DataLocation16::Reg(Reg::DS)),
			0x20 ... 0x25 => Inst::Arithmetic(ArithmeticMode::And, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x26 => {
				self.override_default_segment = Some(Reg::ES);
				let inst = self.parse_instruction();
				self.override_default_segment = None;
				inst
			}
			// DAA
			0x27 => Inst::DecimalAdjustAfter(Reg::AX, RegHalf::Low, AdjustMode::Addition),
			0x28 ... 0x2d => Inst::Arithmetic(ArithmeticMode::Sub, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x2e => {
				self.override_default_segment = Some(Reg::CS);
				let inst = self.parse_instruction();
				self.override_default_segment = None;
				inst
			}
			0x2f => Inst::DecimalAdjustAfter(Reg::AX, RegHalf::Low, AdjustMode::Subtraction),
			0x30 ... 0x35 => Inst::Arithmetic(ArithmeticMode::Xor, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x36 => {
				self.override_default_segment = Some(Reg::SS);
				let inst = self.parse_instruction();
				self.override_default_segment = None;
				inst
			}
			0x37 => Inst::AsciiAdjustAfter(Reg::AX, AdjustMode::Addition),
			0x38 ... 0x3d => Inst::Arithmetic(ArithmeticMode::Cmp, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x3c => {
				let imm = self.read_ip_u8();
				Inst::Arithmetic(ArithmeticMode::Cmp, SourceDestination::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low), DataLocation8::Immediate(imm)))
			}
			0x3d => {
				let imm = self.read_ip_u16();
				Inst::Arithmetic(ArithmeticMode::Cmp, SourceDestination::Size16(DataLocation16::Reg(Reg::AX), DataLocation16::Immediate(imm)))
			}
			0x3e => {
				self.override_default_segment = Some(Reg::DS);
				let inst = self.parse_instruction();
				self.override_default_segment = None;
				inst
			}
			0x40 ... 0x47 => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				//Inst::IncBy16(DataLocation16::Reg(reg), 1)
				Inst::Inc(DataLocation::Size16(DataLocation16::Reg(reg)))
			}
			0x48 ... 0x4f => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				//Inst::DecBy16(DataLocation16::Reg(reg), 1)
				Inst::Dec(DataLocation::Size16(DataLocation16::Reg(reg)))
			}
			0x50 ... 0x57 => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				Inst::Push16(DataLocation16::Reg(reg))
			}
			0x58 ... 0x5f => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				Inst::Pop16(DataLocation16::Reg(reg))
			}
			0x60 => Inst::PushAllGeneralPurposeRegisters,
			0x61 => Inst::PopAllGeneralPurposeRegisters,
			0x70 ... 0x7f => {
				let double_condition = opcode - 0x70;
				let condition_type = JumpConditionType::from_u8(double_condition >> 1).unwrap();
				let negate = double_condition % 2 == 1;
				let offset = treat_u8_as_i8(self.read_ip_u8()) as i32;
				Inst::Jump{condition: Some(JumpCondition{condition: condition_type, negate}), offset}
			}
			0x80 | 0x82 => {
				let (mode_index, destination) = self.read_modrm_with_immediate_reg_u8();
				let arithmetic_mode = ArithmeticMode::from_u8(mode_index).unwrap();
				let imm = self.read_ip_u8();
				Inst::Arithmetic(arithmetic_mode, SourceDestination::Size8(DataLocation8::Immediate(imm), destination))
			}
			0x81 => {
				let (mode_index, destination) = self.read_modrm_with_immediate_reg_u16();
				let arithmetic_mode = ArithmeticMode::from_u16(mode_index).unwrap();
				let imm = self.read_ip_u16();
				Inst::Arithmetic(arithmetic_mode, SourceDestination::Size16(DataLocation16::Immediate(imm), destination))
			}
			// 0x82: See 0x80
			0x83 => {
				let (mode_index, destination) = self.read_modrm_with_immediate_reg_u16();
				let arithmetic_mode = ArithmeticMode::from_u16(mode_index).unwrap();
				let raw_imm = self.read_ip_u8();
				// Read a u8 then sign-extend it to u16:
				let imm = treat_i16_as_u16(treat_u8_as_i8(raw_imm) as i16);
				//println!("0x83 {}, {}", raw_imm, imm);
				Inst::Arithmetic(arithmetic_mode, SourceDestination::Size16(DataLocation16::Immediate(imm as u16), destination))
			}
			// XCHG - swap register/memory with register
			0x86 | 0x87 => Inst::Swap(self.read_standard_source_destination(opcode, ModRmRegMode::Reg)),
			0x88 ... 0x8b => Inst::Mov(self.read_standard_source_destination(opcode, ModRmRegMode::Reg)),
			0x8c => Inst::Mov(self.read_modrm_source_destination(OpSize::Size16, OpDirection::Source, ModRmRegMode::Seg)),
			// LEA
			0x8d => {
				let source_destination = self.read_modrm_source_destination(OpSize::Size16, OpDirection::Destination, ModRmRegMode::Reg);
				if let SourceDestination::Size16(DataLocation16::Memory{address16, ..}, destination) = source_destination {
					Inst::Mov(SourceDestination::Size16(DataLocation16::Address16(address16), destination))
				} else {
					panic!("Expected Memory as the source {:?}", source_destination);
				}
			}
			// MOV
			// https://www.felixcloutier.com/x86/mov
			// The SS register is weird.
			0x8e => Inst::Mov(self.read_modrm_source_destination(OpSize::Size16, OpDirection::Destination, ModRmRegMode::Seg)),
			// XCHG
			0x90 ... 0x97 => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				Inst::Swap(SourceDestination::Size16(DataLocation16::Reg(reg), DataLocation16::Reg(Reg::AX)))
			}
			// CBW - Convert byte to word via sign extension. Ie. get a i8 and turn it into a u16.
			0x98 => Inst::SignExtend8To16{source: DataLocation8::Reg(Reg::AX, RegHalf::Low), destination: DataLocation16::Reg(Reg::AX)},
			// CWD - Convert word to double.
			0x99 => Inst::SignExtend16To32{source: DataLocation16::Reg(Reg::AX), destination_low: DataLocation16::Reg(Reg::AX), destination_high: DataLocation16::Reg(Reg::DX)},
			// CALLF
			0x9a => {
				let ip = self.read_ip_u16();
				let cs = self.read_ip_u16();
				Inst::CallAbsolute{ip, cs}
			}
			// PUSHF (push flags register)
			0x9c => Inst::Push16(DataLocation16::Reg(Reg::Flags)),
			// POPF
			0x9d => Inst::PopFlags,
			// MOV (moffs8 -> AX)
			0xa0 => {
				let offset = self.read_ip_u16();
				Inst::Mov(SourceDestination::Size8(DataLocation8::Memory{seg: self.resolve_default_segment(Reg::DS), address16: Address16::Immediate(offset)}, DataLocation8::Reg(Reg::AX, RegHalf::Low)))
			}
			// MOV (moffs16 -> AX)
			0xa1 => {
				let offset = self.read_ip_u16();
				Inst::Mov(SourceDestination::Size16(DataLocation16::Memory{seg: self.resolve_default_segment(Reg::DS), address16: Address16::Immediate(offset)}, DataLocation16::Reg(Reg::AX)))
			}
			0xa2 => {
				let addr = self.read_ip_u16();
				Inst::Mov(SourceDestination::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low), DataLocation8::Memory{seg: self.resolve_default_segment(Reg::DS), address16: Address16::Immediate(addr)}))
			}
			0xa3 => {
				let addr = self.read_ip_u16();
				Inst::Mov(SourceDestination::Size16(DataLocation16::Reg(Reg::AX), DataLocation16::Memory{seg: self.resolve_default_segment(Reg::DS), address16: Address16::Immediate(addr)}))
			}
			// MOVSB (the ES segment cannot be overridden)
			// "string" means that it increments (or decrements if the direction flag is set) the
			// memory address register(s) after doing an operation.
			0xa4 => Inst::MovAndIncrement(SourceDestination::Size8(DataLocation8::Memory{seg: self.resolve_default_segment(Reg::DS), address16: Address16::Reg(Reg::SI)}, DataLocation8::Memory{seg: Reg::ES, address16: Address16::Reg(Reg::DI)}), Reg::SI, Some(Reg::DI)),
			// MOVSW (the ES segment cannot be overridden)
			0xa5 => Inst::MovAndIncrement(SourceDestination::Size16(DataLocation16::Memory{seg: self.resolve_default_segment(Reg::DS), address16: Address16::Reg(Reg::SI)}, DataLocation16::Memory{seg: Reg::ES, address16: Address16::Reg(Reg::DI)}), Reg::SI, Some(Reg::DI)),
			// TEST
			0xa8 => {
				let imm = self.read_ip_u8();
				Inst::BitwiseCompareWithAnd(SourceDestination::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low), DataLocation8::Immediate(imm)))
			}
			0xa9 => {
				let imm = self.read_ip_u16();
				Inst::BitwiseCompareWithAnd(SourceDestination::Size16(DataLocation16::Reg(Reg::AX), DataLocation16::Immediate(imm)))
			}
			// STOSB (the ES segment cannot be overridden)
			0xaa => Inst::MovAndIncrement(SourceDestination::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low), DataLocation8::Memory{seg: Reg::ES, address16: Address16::Reg(Reg::DI)}), Reg::DI, None),
			// STOSW (the ES segment cannot be overridden)
			0xab => Inst::MovAndIncrement(SourceDestination::Size16(DataLocation16::Reg(Reg::AX), DataLocation16::Memory{seg: Reg::ES, address16: Address16::Reg(Reg::DI)}), Reg::DI, None),
			// LODSB
			0xac => Inst::MovAndIncrement(SourceDestination::Size8(DataLocation8::Memory{seg: self.resolve_default_segment(Reg::DS), address16: Address16::Reg(Reg::SI)}, DataLocation8::Reg(Reg::AX, RegHalf::Low)), Reg::SI, None),
			// LODSW
			0xad => Inst::MovAndIncrement(SourceDestination::Size16(DataLocation16::Memory{seg: self.resolve_default_segment(Reg::DS), address16: Address16::Reg(Reg::SI)}, DataLocation16::Reg(Reg::AX)), Reg::SI, None),
			0xb0 ... 0xb7 => {
				let (reg, reg_half) = Reg::reg8((opcode & 0b111) as usize).unwrap();
				let imm = self.read_ip_u8();
				Inst::Mov(SourceDestination::Size8(DataLocation8::Immediate(imm), DataLocation8::Reg(reg, reg_half)))
			}
			0xb8 ... 0xbf => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				let imm = self.read_ip_u16();
				Inst::Mov(SourceDestination::Size16(DataLocation16::Immediate(imm), DataLocation16::Reg(reg)))
			}
			0xc0 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u8();
				let shift_mode = ShiftMode::from_u8(shift_index).unwrap();
				let imm = self.read_ip_u8();
				Inst::Rotate{by: DataLocation8::Immediate(imm), target: DataLocation::Size8(destination), mode: shift_mode}
			}
			0xc1 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u16();
				let shift_mode = ShiftMode::from_u16(shift_index).unwrap();
				let imm = self.read_ip_u8();
				Inst::Rotate{by: DataLocation8::Immediate(imm), target: DataLocation::Size16(destination), mode: shift_mode}
			}
			// RETN
			0xc2 => {
				let extra_pop = self.read_ip_u16();
				Inst::Ret{extra_pop}
			}
			// RETN
			0xc3 => {
				Inst::Ret{extra_pop: 0}
			}
			// LES | LDS (load absolute pointer in the ES/DS segment)
			0xc4 | 0xc5 => {
				let source_destination = self.read_modrm_source_destination(OpSize::Size16, OpDirection::Destination, ModRmRegMode::Reg);
				if let SourceDestination::Size16(DataLocation16::Memory{seg, address16: Address16::DisplacementRel(displacement, offset)}, DataLocation16::Reg(out_reg_l)) = source_destination {
					let out_reg_h = if opcode == 0xc4 { Reg::ES } else { Reg::DS };
					Inst::Load32{seg, address16: Address16::DisplacementRel(displacement, offset), out_reg_h, out_reg_l}
				} else {
					panic!("Expected source to be Memory/DisplacementRel and destination to be Reg: {:?}", source_destination);
				}
			}
			0xc6 => {
				let (_, destination) = self.read_modrm_source_destination(OpSize::Size8, OpDirection::Source, ModRmRegMode::Reg).split();
				let imm = self.read_ip_u8();
				Inst::Mov(destination.with_immediate_source(imm as u16))
			}
			0xc7 => {
				// TODO: 0xc7  698084
				let (_, destination) = self.read_modrm_source_destination(OpSize::Size16, OpDirection::Source, ModRmRegMode::Reg).split();
				let imm = self.read_ip_u16();
				Inst::Mov(destination.with_immediate_source(imm))
			}
			0xca => {
				let extra_pop = self.read_ip_u16();
				Inst::RetAbsolute{extra_pop}
			}
			0xcb => Inst::RetAbsolute{extra_pop: 0},
			0xcd => {
				let interrupt_index = self.read_ip_u8();
				Inst::Interrupt(interrupt_index)
			}
			0xce => {
				// INTO calls the overflow exception, which is #4
				Inst::InterruptIf(4, Flag::Overflow)
			}
			// IRET
			0xcf => Inst::ReturnFromInterrupt,
			0xd0 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u8();
				let shift_mode = ShiftMode::from_u8(shift_index).unwrap();
				Inst::Rotate{by: DataLocation8::Immediate(1), target: DataLocation::Size8(destination), mode: shift_mode}
			}
			0xd1 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u16();
				let shift_mode = ShiftMode::from_u16(shift_index).unwrap();
				Inst::Rotate{by: DataLocation8::Immediate(1), target: DataLocation::Size16(destination), mode: shift_mode}
			}
			0xd2 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u8();
				let shift_mode = ShiftMode::from_u8(shift_index).unwrap();
				Inst::Rotate{by: DataLocation8::Reg(Reg::CX, RegHalf::Low), target: DataLocation::Size8(destination), mode: shift_mode}
			}
			0xd3 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u16();
				let shift_mode = ShiftMode::from_u16(shift_index).unwrap();
				Inst::Rotate{by: DataLocation8::Reg(Reg::CX, RegHalf::Low), target: DataLocation::Size16(destination), mode: shift_mode}
			}
			// AAD/ADX
			0xd5 => {
				let base = self.read_ip_u8();
				Inst::CombineBytesAsNumberWithBase(Reg::AX, base)
			}
			// LOOP
			0xe2 => {
				let offset = treat_u8_as_i8(self.read_ip_u8()) as i32;
				Inst::JumpAndDecrementUntilZero{offset, dec_reg: Reg::CX}
			}
			0xe3 => {
				let offset = treat_u8_as_i8(self.read_ip_u8()) as i32;
				Inst::JumpZeroReg{offset, reg: Reg::CX}
			}
			// IN
			0xe4 => {
				let port_index = self.read_ip_u8() as u16;
				Inst::PortInput{port_index: DataLocation16::Immediate(port_index), destination: DataLocation::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low))}
			}
			0xe5 => {
				let port_index = self.read_ip_u8() as u16;
				Inst::PortInput{port_index: DataLocation16::Immediate(port_index), destination: DataLocation::Size16(DataLocation16::Reg(Reg::AX))}
			}
			0xe6 => {
				let port_index = self.read_ip_u8() as u16;
				Inst::PortOutput{port_index: DataLocation16::Immediate(port_index), source: DataLocation::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low))}
			}
			0xe7 => {
				let port_index = self.read_ip_u8() as u16;
				Inst::PortOutput{port_index: DataLocation16::Immediate(port_index), source: DataLocation::Size16(DataLocation16::Reg(Reg::AX))}
			}
			0xe8 => {
				let offset = treat_u16_as_i16(self.read_ip_u16()) as i32;
				Inst::Call(offset)
			}
			0xe9 => {
				let offset = treat_u16_as_i16(self.read_ip_u16()) as i32;
				Inst::Jump{condition: None, offset}
			}
			0xeb => {
				let offset = treat_u8_as_i8(self.read_ip_u8()) as i32;
				Inst::Jump{condition: None, offset}
			}
			0xec => {
				Inst::PortInput{port_index: DataLocation16::Reg(Reg::DX), destination: DataLocation::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low))}
			}
			0xee => {
				Inst::PortOutput{port_index: DataLocation16::Reg(Reg::DX), source: DataLocation::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low))}
			}
			0xef => {
				Inst::PortOutput{port_index: DataLocation16::Reg(Reg::DX), source: DataLocation::Size16(DataLocation16::Reg(Reg::AX))}
			}
			0xf0 => {
				// LOCK: This is supposed to set the LOCK pin on the CPU to ON for the duration of
				// the next instruction's execution.
 				Inst::NoOp
			}
			// REPZ
			0xf3 => Inst::RepeatNextRegTimes{reg: Reg::CX, until_zero_flag: Some(true)},
			0xf4 => Inst::Halt,
			// CMC (compliment (flip the state of the) carry flag)
			0xf5 => Inst::InvertFlag(Flag::Carry),
			0xf6 => {
				let (inst_index, destination) = self.read_modrm_with_immediate_reg_u8();
				match inst_index {
					0 => {
						let imm = self.read_ip_u8();
						Inst::BitwiseCompareWithAnd(SourceDestination::Size8(DataLocation8::Immediate(imm), destination))
					}
					2 => Inst::Negate(DataLocation::Size8(destination)),
					3 => Inst::NegateSigned(DataLocation::Size8(destination)),
					4...7 => Inst::ScalarOperation8{mode: self.get_standard_scalar_mode(inst_index as u8), value_quot_rem: Reg::AX, by: destination},
					_ => {
						//println!("MEM: {:?}", &self.memory[0xb8000..0xb8000+0x1000]);
						panic!("Unknown 0xf6 mode: 0x{:x}", inst_index)
					}
				}
			}
			0xf7 => {
				let (inst_index, destination) = self.read_modrm_with_immediate_reg_u16();
				match inst_index {
					0 => {
						let imm = self.read_ip_u16();
						Inst::BitwiseCompareWithAnd(SourceDestination::Size16(DataLocation16::Immediate(imm), destination))
					}
					2 => Inst::Negate(DataLocation::Size16(destination)),
					3 => Inst::NegateSigned(DataLocation::Size16(destination)),
					4...7 => Inst::ScalarOperation16{mode: self.get_standard_scalar_mode(inst_index as u8), value_low_quot: Reg::AX, value_high_rem: Reg::DX, by: destination},
					_ => panic!("Unknown 0xf7 mode: 0x{:x}", inst_index)
				}
			}
			0xf8 => Inst::SetFlag(Flag::Carry, false),
			0xf9 => Inst::SetFlag(Flag::Carry, true),
			0xfa => Inst::SetFlag(Flag::Interrupt, false),
			0xfb => Inst::SetFlag(Flag::Interrupt, true),
			0xfc => Inst::SetFlag(Flag::Direction, false),
			0xfd => Inst::SetFlag(Flag::Direction, true),
			0xfe => {
				let (inst_index, destination) = self.read_modrm_with_immediate_reg_u8();
				match inst_index {
					0 => Inst::Inc(DataLocation::Size8(destination)),
					1 => Inst::Dec(DataLocation::Size8(destination)),
					_ => panic!("Unknown 0xfe mode: 0x{:x}", inst_index)
				}
			}
			0xff => {
				let (inst_index, destination) = self.read_modrm_with_immediate_reg_u16();
				match inst_index {
					0 => {
						Inst::Inc(DataLocation::Size16(destination))
					}
					3 => {
						if let DataLocation16::Memory{seg, address16} = destination {
							Inst::CallAbsoluteWithAddress{seg, address16}
						} else {
							panic!("Expected Memory for CALLF: {:?}", destination);
						}
					}
					5 => {
						// JMPF (jump far)
						// This gets the modrm destination (which should be a seg/offset for getting
						// a memory address) and reads 32 bits from that position. The first 16 bits
						// go into the IP and the second 16 go into the CS.
						if let DataLocation16::Memory{seg, address16} = destination {
							Inst::JumpAbsolute{seg, address16}
						} else {
							panic!("Expected Memory for JMPF: {:?}", destination);
						}
					}
					6 => {
						Inst::Push16(destination)
					}
					_ => panic!("Unknown 0xff mode: 0x{:x}", inst_index)
				}
			}
			_ => {
				//println!("{}", String::from_utf8_lossy(&self.memory));
				panic!("Unknown opcode: 0x{:02x} (Parsing instruction #{})", opcode, self.number_of_parsed_instructions);
			}
		}
	}
	
	/// Returns a value if the destination is meant to be overwritten (eg. a cmp doesn't overwrite
	/// the destination, so it would return None).
	fn apply_arithmetic_inst<ValueType: AnyUnsignedInt8086>(&mut self, value1: ValueType, value2: ValueType, mode: ArithmeticMode, change_carry_flag: bool) -> Option<ValueType> {
		enum CarryType {
			Clear,
			Add,
			Sub,
			SubWithBorrow,
		}
		let (result_value, should_write, set_adjust_flag, carry_type) = match mode {
			ArithmeticMode::Add => (value1.wrapping_add(value2), true, true, CarryType::Add),
			ArithmeticMode::Or => (value1 | value2, true, false, CarryType::Clear),
			ArithmeticMode::AddWithCarry => {
				let carry_value = if self.get_flag(Flag::Carry) { ValueType::ONE } else { ValueType::ZERO };
				(value1.wrapping_add(value2).wrapping_add(carry_value), true, true, CarryType::Add)
			}
			ArithmeticMode::SubWithBorrow => {
				let carry_value = if self.get_flag(Flag::Carry) { ValueType::ONE } else { ValueType::ZERO };
				(value1.wrapping_sub(value2.wrapping_add(carry_value)), true, true, CarryType::SubWithBorrow)
			}
			ArithmeticMode::And => (value1 & value2, true, false, CarryType::Clear),
			ArithmeticMode::Sub => (value1.wrapping_sub(value2), true, true, CarryType::Sub),
			ArithmeticMode::Xor => (value1 ^ value2, true, false, CarryType::Clear),
			ArithmeticMode::Cmp => (value1.wrapping_sub(value2), false, true, CarryType::Sub),
		};
		
		self.set_standard_zero_sign_partiy_flags(result_value);
		
		if set_adjust_flag {
			self.set_flag(Flag::Adjust, (((value1 ^ value2) ^ result_value) & (ValueType::ONE << 4)) != ValueType::ZERO);
		} else {
			// TODO: Is this supposed to be here? (specifically for XOR/opcode 0x33)
			//self.set_flag(Flag::Adjust, false);
		}
		
		let sign_bit = ValueType::ONE << (ValueType::BIT_SIZE - 1);
		
		match carry_type {
			CarryType::Clear => {
				self.set_flag(Flag::Overflow, false);
				self.set_flag(Flag::Carry, false);
			}
			CarryType::Add => {
				self.set_flag(Flag::Overflow, (((value1 ^ value2 ^ sign_bit) & (result_value ^ value2)) & sign_bit) != ValueType::ZERO);
				// If the result value is supposed to increase but it decreases, the carry flag is set.
				if change_carry_flag { self.set_flag(Flag::Carry, result_value < value1); }
			}
			CarryType::Sub => {
				self.set_flag(Flag::Overflow, (((value1 ^ value2) & (value1 ^ result_value)) & sign_bit) != ValueType::ZERO);
				if change_carry_flag { self.set_flag(Flag::Carry, value1 < value2); }
			}
			CarryType::SubWithBorrow => {
				self.set_flag(Flag::Overflow, (((value1 ^ value2) & (value1 ^ result_value)) & sign_bit) != ValueType::ZERO);
				if change_carry_flag { self.set_flag(Flag::Carry, value1 < result_value); }
			}
		}
		
		if should_write { Some(result_value) } else { None }
	}
	
	fn apply_rotate_inst<ValueType: AnyUnsignedInt8086>(&mut self, rotate_amount: u8, mut value: ValueType, mode: ShiftMode) -> ValueType {
		// https://www.felixcloutier.com/x86/rcl:rcr:rol:ror
		// https://www.felixcloutier.com/x86/sal:sar:shl:shr
		let masked_rotate_amount = rotate_amount & 0b11111;
		let real_rotate_amount = (masked_rotate_amount) % ValueType::BIT_SIZE;
		let carry_rotate_amount = (masked_rotate_amount) % (ValueType::BIT_SIZE + 1);
		match mode {
			ShiftMode::RotateLeft => { // ROL
				value = value.rotate_left(real_rotate_amount as u32);
				if masked_rotate_amount != 0 {
					self.set_flag(Flag::Carry, value.least_significant_bit(0));
				}
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, value.most_significant_bit(0) != value.least_significant_bit(0));
				}
			}
			ShiftMode::RotateRight => { // ROR
				value = value.rotate_right(real_rotate_amount as u32);
				if masked_rotate_amount != 0 {
					self.set_flag(Flag::Carry, value.most_significant_bit(0));
				}
				if masked_rotate_amount == 1 {
					// The most signficant bit != the second most significant bit.
					self.set_flag(Flag::Overflow, value.most_significant_bit(0) != value.most_significant_bit(1));
				}
			}
			ShiftMode::RotateLeftWithCarry => { // RCL
				for _ in 0..carry_rotate_amount {
					let original_most_sig_bit = value.most_significant_bit(0);
					value = value << 1;
					if self.get_flag(Flag::Carry) {
						value += ValueType::ONE;
					}
					self.set_flag(Flag::Carry, original_most_sig_bit);
				}
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, value.most_significant_bit(0) != self.get_flag(Flag::Carry));
				}
			}
			ShiftMode::RotateRightWithCarry => { // RCR
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, value.most_significant_bit(0) != self.get_flag(Flag::Carry));
				}
				for _ in 0..carry_rotate_amount {
					let original_least_sig_bit = value.least_significant_bit(0);
					value = value >> 1;
					if self.get_flag(Flag::Carry) {
						value |= ValueType::ONE << (ValueType::BIT_SIZE - 1);
					}
					self.set_flag(Flag::Carry, original_least_sig_bit);
				}
			}
			ShiftMode::ShiftLeft | ShiftMode::ShiftLeftArithmethic => { // SHL
				if rotate_amount != 0 {
					value = value << rotate_amount - 1;
					self.set_flag(Flag::Carry, value.most_significant_bit(0));
					value = value << 1;
				}
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, value.most_significant_bit(0) != self.get_flag(Flag::Carry));
				}
				self.set_flag(Flag::Adjust, (rotate_amount & 0x1f) != 0);
				self.set_standard_zero_sign_partiy_flags(value);
			}
			ShiftMode::ShiftRight => { // SHR
				let original_value = value;
				if rotate_amount != 0 {
					value = value >> rotate_amount - 1;
					self.set_flag(Flag::Carry, value.least_significant_bit(0));
					value = value >> 1;
				}
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, original_value.most_significant_bit(0));
				}
				self.set_standard_zero_sign_partiy_flags(value);
			}
			// SAL: ShiftMode::ShiftLeftArithmethic is the same as ShiftMode::ShiftLeft
			ShiftMode::ShiftRightArithmethic => { // SAR
				if rotate_amount != 0 {
					value = value >> rotate_amount - 1;
					self.set_flag(Flag::Carry, value.least_significant_bit(0));
					value = value >> 1;
				}
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, false);
				}
				self.set_standard_zero_sign_partiy_flags(value);
			}
		};
		
		value
	}
	
	fn set_standard_zero_sign_partiy_flags<ValueType: AnyUnsignedInt8086>(&mut self, value: ValueType) {
		self.set_flag(Flag::Zero, value == ValueType::ZERO);
		self.set_flag(Flag::Sign, value.most_significant_bit(0));
		self.set_flag(Flag::Parity, u8_on_bits_are_odd(value.least_significant_byte()));
	}
	
	fn interrupt(&mut self, interrupt_index: u8) {
		println!("Interrupt: {:?}", interrupt_index);
		let interrupt_addr = interrupt_index as u32 * INTERRUPT_TABLE_ENTRY_BYTES as u32;
		let ip = self.peek_u16(interrupt_addr);
		let cs = self.peek_u16(interrupt_addr + 2);
		
		self.push_u16(self.get_reg_u16(Reg::Flags));
		self.push_u16(self.get_reg_u16(Reg::CS));
		self.push_u16(self.get_reg_u16(Reg::IP));
		
		self.set_reg_u16(Reg::IP, ip);
		self.set_reg_u16(Reg::CS, cs);
		self.halted = false;
		self.set_flag(Flag::Interrupt, false);
	}
	
	fn return_from_interrupt(&mut self) {
		let ip = self.pop_u16();
		let cs = self.pop_u16();
		let old_flags = self.pop_u16();
		let interrupt_flag_bit = 0b1 << (Flag::Interrupt as u16);
		let old_interrupt_flag = (old_flags & interrupt_flag_bit) != 0;
		
		self.set_reg_u16(Reg::IP, ip);
		self.set_reg_u16(Reg::CS, cs);
		
		self.set_flag(Flag::Interrupt, old_interrupt_flag);
	}
	
	pub fn interrupt_on_next_step(&mut self, interrupt_index: u8) {
		self.pending_interrupts.push_back(interrupt_index);
	}
	
	pub fn apply_instruction(&mut self, inst: &Inst, event_handler: &mut EventHandler) {
		match *inst {
			Inst::NoOp => {}
			Inst::Push16(ref location) => {
				self.push_u16(self.get_data_u16(location));
			}
			Inst::Pop16(ref location) => {
				let value = self.pop_u16();
				self.set_data_u16(location, value);
			}
			Inst::PopFlags => {
				let mut value = self.pop_u16();
				// TODO: ZETA sets these bits, which seems wrong.
				value |= 0xf002;
				self.set_reg_u16(Reg::Flags, value);
			}
			Inst::PushAllGeneralPurposeRegisters => {
				let start_sp = self.get_reg_u16(Reg::SP);
				self.push_u16(self.get_reg_u16(Reg::AX));
				self.push_u16(self.get_reg_u16(Reg::CX));
				self.push_u16(self.get_reg_u16(Reg::DX));
				self.push_u16(self.get_reg_u16(Reg::BX));
				self.push_u16(start_sp);
				self.push_u16(self.get_reg_u16(Reg::BP));
				self.push_u16(self.get_reg_u16(Reg::SI));
				self.push_u16(self.get_reg_u16(Reg::DI));
			}
			Inst::PopAllGeneralPurposeRegisters => {
				self.registers[Reg::DI as usize] = self.pop_u16();
				self.registers[Reg::SI as usize] = self.pop_u16();
				self.registers[Reg::BP as usize] = self.pop_u16();
				let _sp = self.pop_u16();
				self.registers[Reg::BX as usize] = self.pop_u16();
				self.registers[Reg::DX as usize] = self.pop_u16();
				self.registers[Reg::CX as usize] = self.pop_u16();
				self.registers[Reg::AX as usize] = self.pop_u16();
			}
			Inst::Mov(SourceDestination::Size8(ref source, ref destination)) => {
				let value = self.get_data_u8(&source);
				self.set_data_u8(&destination, value);
			}
			Inst::Mov(SourceDestination::Size16(ref source, ref destination)) => {
				let value = self.get_data_u16(&source);
				self.set_data_u16(&destination, value);
				if let DataLocation16::Reg(Reg::SS) = destination {
					// https://www.felixcloutier.com/x86/mov
					// Loading the SS register with MOV skips interrupts, so just manually execute another instruction to force them to be skipped.
					// TODO: Move this into a machine state.
					let next_inst = self.parse_instruction();
					self.apply_instruction(&next_inst, event_handler);
				}
			}
			Inst::Load32{seg, ref address16, out_reg_h, out_reg_l} => {
				let addr = self.get_seg_offset(seg, self.calculate_effective_address16(&address16));
				let in_h = self.peek_u16(addr + 2);
				let in_l = self.peek_u16(addr);
				self.set_reg_u16(out_reg_h, in_h);
				self.set_reg_u16(out_reg_l, in_l);
			}
			Inst::Swap(SourceDestination::Size8(ref left, ref right)) => {
				let value_left = self.get_data_u8(&left);
				let value_right = self.get_data_u8(&right);
				self.set_data_u8(&left, value_right);
				self.set_data_u8(&right, value_left);
			}
			Inst::Swap(SourceDestination::Size16(ref left, ref right)) => {
				let value_left = self.get_data_u16(&left);
				let value_right = self.get_data_u16(&right);
				self.set_data_u16(&left, value_right);
				self.set_data_u16(&right, value_left);
			}
			Inst::MovAndIncrement(SourceDestination::Size8(ref source, ref destination), inc_reg, inc_reg2) => {
				let value = self.get_data_u8(&source);
				self.set_data_u8(&destination, value);
				if self.get_flag(Flag::Direction) {
					self.sub_from_reg(inc_reg, 1);
					if let Some(inc_reg2) = inc_reg2 { self.sub_from_reg(inc_reg2, 1); }
				} else {
					self.add_to_reg(inc_reg, 1);
					if let Some(inc_reg2) = inc_reg2 { self.add_to_reg(inc_reg2, 1); }
				}
			}
			Inst::MovAndIncrement(SourceDestination::Size16(ref source, ref destination), inc_reg, inc_reg2) => {
				let value = self.get_data_u16(&source);
				self.set_data_u16(&destination, value);
				if self.get_flag(Flag::Direction) {
					self.sub_from_reg(inc_reg, 2);
					if let Some(inc_reg2) = inc_reg2 { self.sub_from_reg(inc_reg2, 2); }
				} else {
					self.add_to_reg(inc_reg, 2);
					if let Some(inc_reg2) = inc_reg2 { self.add_to_reg(inc_reg2, 2); }
				}
			}
			Inst::Arithmetic(mode, SourceDestination::Size8(ref source, ref destination)) => {
				let value1 = self.get_data_u8(&destination);
				let value2 = self.get_data_u8(&source);
				if let Some(result_value) = self.apply_arithmetic_inst(value1, value2, mode, true) {
					self.set_data_u8(&destination, result_value);
				}
			}
			Inst::Arithmetic(mode, SourceDestination::Size16(ref source, ref destination)) => {
				let value1 = self.get_data_u16(&destination);
				let value2 = self.get_data_u16(&source);
				if let Some(result_value) = self.apply_arithmetic_inst(value1, value2, mode, true) {
					self.set_data_u16(&destination, result_value);
				}
			}
			Inst::ScalarOperation8{mode, value_quot_rem, ref by} => {
				match mode {
					ScalarMode::Mul => {
						let value = self.get_reg_u8(value_quot_rem, RegHalf::Low);
						let by_value = self.get_data_u8(by);
						let result = (value as u16).wrapping_mul(by_value as u16);
						self.set_reg_u16(value_quot_rem, result);
						let bigger_than_u8 = result > 0xff;
						self.set_flag(Flag::Overflow, bigger_than_u8);
						self.set_flag(Flag::Carry, bigger_than_u8);
					}
					ScalarMode::Div => {
						let value = self.get_reg_u16(value_quot_rem);
						let divisor = self.get_data_u8(by);
						if divisor == 0 {
							// Divide by zero
							self.interrupt_on_next_step(0);
						} else {
							let quotient = value / (divisor as u16);
							let quotient8 = (quotient & 0xff) as u8;
							let remainder = (value % (divisor as u16)) as u8;
							if quotient != quotient8 as u16 {
								// Result doesn't fit
								self.interrupt_on_next_step(0);
							} else {
								self.set_reg_u8(value_quot_rem, RegHalf::High, remainder);
								self.set_reg_u8(value_quot_rem, RegHalf::Low, quotient8);
							}
						}
					}
					_ => panic!("Unimplemented: {:?}", mode)
				}
			}
			Inst::ScalarOperation16{mode, value_low_quot, value_high_rem, ref by} => {
				match mode {
					ScalarMode::Mul => {
						let value = self.get_reg_u16(value_low_quot);
						let by_value = self.get_data_u16(by);
						let result = (value as u32).wrapping_mul(by_value as u32);
						self.set_reg_u16(value_low_quot, (result & 0xffff) as u16);
						self.set_reg_u16(value_high_rem, ((result & 0xffff0000) >> 16) as u16);
						let bigger_than_u16 = result > 0xffff;
						self.set_flag(Flag::Overflow, bigger_than_u16);
						self.set_flag(Flag::Carry, bigger_than_u16);
					}
					ScalarMode::SignedMul => {
						let value = treat_u16_as_i16(self.get_reg_u16(value_low_quot));
						let by_value = treat_u16_as_i16(self.get_data_u16(by));
						let signed_result = (value as i32).wrapping_mul(by_value as i32);
						let result = treat_i32_as_u32(signed_result);
						let result_low = (result & 0xffff) as u16;
						self.set_reg_u16(value_low_quot, result_low);
						self.set_reg_u16(value_high_rem, ((result & 0xffff0000) >> 16) as u16);
						let bigger_than_i16 = treat_u16_as_i16(result_low) as i32 != signed_result;
						self.set_flag(Flag::Overflow, bigger_than_i16);
						self.set_flag(Flag::Carry, bigger_than_i16);
					}
					ScalarMode::Div => {
						let value = self.get_reg_u16(value_low_quot) as u32 + ((self.get_reg_u16(value_high_rem) as u32) << 16);
						let divisor = self.get_data_u16(by);
						if divisor == 0 {
							// Divide by zero
							self.interrupt_on_next_step(0);
						} else {
							let quotient = value / (divisor as u32);
							let quotient16 = (quotient & 0xffff) as u16;
							if quotient != quotient16 as u32 {
								// Result doesn't fit
								self.interrupt_on_next_step(0);
							} else {
								let remainder = (value % (divisor as u32)) as u16;
								self.set_reg_u16(value_low_quot, quotient16);
								self.set_reg_u16(value_high_rem, remainder);
							}
						}
					}
					ScalarMode::SignedDiv => {
						// TODO
						let value = treat_u32_as_i32(self.get_reg_u16(value_low_quot) as u32 + ((self.get_reg_u16(value_high_rem) as u32) << 16));
						let divisor = treat_u16_as_i16(self.get_data_u16(by));
						if divisor == 0 {
							// Divide by zero
							self.interrupt_on_next_step(0);
						} else {
							let quotient = value / (divisor as i32);
							if quotient >= 0x8000 || quotient < -0x8000 {
								// Result doesn't fit
								self.interrupt_on_next_step(0);
							} else {
								let unsigned_quotient16 = treat_i16_as_u16(quotient as i16);
								let remainder = treat_i16_as_u16((value % (divisor as i32)) as i16);
								self.set_reg_u16(value_low_quot, unsigned_quotient16);
								self.set_reg_u16(value_high_rem, remainder);
							}
						}
					}
					_ => panic!("Unimplemented: {:?}", mode)
				}
			}
			Inst::Negate(DataLocation::Size8(ref target)) => {
				let value = self.get_data_u8(&target);
				self.set_data_u8(&target, !value);
			}
			Inst::Negate(DataLocation::Size16(ref target)) => {
				let value = self.get_data_u16(&target);
				self.set_data_u16(&target, !value);
			}
			Inst::NegateSigned(DataLocation::Size8(ref target)) => {
				let value = self.get_data_u8(&target);
				self.set_flag(Flag::Carry, value != 0);
				let neg = treat_i8_as_u8(-treat_u8_as_i8(value));
				self.set_data_u8(&target, neg);
			}
			Inst::NegateSigned(DataLocation::Size16(ref target)) => {
				let value = self.get_data_u16(&target);
				self.set_flag(Flag::Carry, value != 0);
				let neg = treat_i16_as_u16(-treat_u16_as_i16(value));
				self.set_data_u16(&target, neg);
			}
			Inst::Inc(DataLocation::Size8(ref target)) => {
				let value = self.get_data_u8(&target);
				if let Some(result_value) = self.apply_arithmetic_inst(value, 1, ArithmeticMode::Add, false) {
					self.set_data_u8(&target, result_value);
				}
			}
			Inst::Inc(DataLocation::Size16(ref target)) => {
				let value = self.get_data_u16(&target);
				if let Some(result_value) = self.apply_arithmetic_inst(value, 1, ArithmeticMode::Add, false) {
					self.set_data_u16(&target, result_value);
				}
			}
			Inst::Dec(DataLocation::Size8(ref target)) => {
				let value = self.get_data_u8(&target);
				if let Some(result_value) = self.apply_arithmetic_inst(value, 1, ArithmeticMode::Sub, false) {
					self.set_data_u8(&target, result_value);
				}
			}
			Inst::Dec(DataLocation::Size16(ref target)) => {
				let value = self.get_data_u16(&target);
				if let Some(result_value) = self.apply_arithmetic_inst(value, 1, ArithmeticMode::Sub, false) {
					self.set_data_u16(&target, result_value);
				}
			}
			Inst::Rotate{ref by, target: DataLocation::Size8(ref target), mode} => {
				let rotate_amount = self.get_data_u8(by);
				let value = self.get_data_u8(&target);
				let new_value = self.apply_rotate_inst(rotate_amount, value, mode);
				self.set_data_u8(&target, new_value);
			}
			Inst::Rotate{ref by, target: DataLocation::Size16(ref target), mode} => {
				let rotate_amount = self.get_data_u8(by);
				let value = self.get_data_u16(&target);
				let new_value = self.apply_rotate_inst(rotate_amount, value, mode);
				self.set_data_u16(&target, new_value);
			}
			Inst::BitwiseCompareWithAnd(SourceDestination::Size8(ref source, ref destination)) => {
				let value1 = self.get_data_u8(&destination);
				let value2 = self.get_data_u8(&source);
				let result_value = value1 & value2;
				self.set_standard_zero_sign_partiy_flags(result_value);
				self.set_flag(Flag::Carry, false);
				self.set_flag(Flag::Overflow, false);
			}
			Inst::BitwiseCompareWithAnd(SourceDestination::Size16(ref source, ref destination)) => {
				let value1 = self.get_data_u16(&destination);
				let value2 = self.get_data_u16(&source);
				let result_value = value1 & value2;
				self.set_standard_zero_sign_partiy_flags(result_value);
				self.set_flag(Flag::Carry, false);
				self.set_flag(Flag::Overflow, false);
			}
			Inst::CombineBytesAsNumberWithBase(reg, base) => {
				let low_byte = self.get_reg_u8(reg, RegHalf::Low);
				let high_byte = self.get_reg_u8(reg, RegHalf::High);
				let new_value = (low_byte + (high_byte * base)) as u16;
				self.set_reg_u16(reg, new_value);
			}
			Inst::AsciiAdjustAfter(reg, mode) => {
				if self.get_reg_u8(reg, RegHalf::Low) & 0x0f > 9 || self.get_flag(Flag::Adjust) {
					match mode {
						AdjustMode::Addition => {
							self.set_reg_u16(reg, self.get_reg_u16(reg).wrapping_add(0x106));
						}
						AdjustMode::Subtraction => {
							self.set_reg_u16(reg, self.get_reg_u16(reg).wrapping_sub(6));
							self.set_reg_u8(reg, RegHalf::High, self.get_reg_u8(reg, RegHalf::High).wrapping_sub(1));
						}
					}
					self.set_flag(Flag::Adjust, true);
					self.set_flag(Flag::Carry, true);
				} else {
					self.set_flag(Flag::Adjust, false);
					self.set_flag(Flag::Carry, false);
				}
				self.set_reg_u8(reg, RegHalf::Low, self.get_reg_u8(reg, RegHalf::Low) & 0x0f);
			}
			Inst::DecimalAdjustAfter(reg, reg_half, mode) => {
				let old_value = self.get_reg_u8(reg, reg_half);
				let old_carry = self.get_flag(Flag::Carry);
				self.set_flag(Flag::Carry, false);
				if old_value & 0x0f > 9 || self.get_flag(Flag::Adjust) {
					match mode {
						AdjustMode::Addition => {
							let new_value = old_value.wrapping_add(6);
							self.set_reg_u8(reg, reg_half, new_value);
							let new_carry = old_carry || new_value < 6;
							self.set_flag(Flag::Carry, new_carry);
						}
						AdjustMode::Subtraction => {
							let new_value = old_value.wrapping_sub(6);
							self.set_reg_u8(reg, reg_half, new_value);
							let new_carry = old_carry || old_value < 6;
							self.set_flag(Flag::Carry, new_carry);
						}
					}
					self.set_flag(Flag::Adjust, true);
				} else {
					self.set_flag(Flag::Adjust, false);
				}
				if old_value > 0x99 || old_carry {
					self.set_reg_u8(reg, reg_half, match mode {
						AdjustMode::Addition => old_value.wrapping_add(0x60),
						AdjustMode::Subtraction => old_value.wrapping_sub(0x60),
					});
					self.set_flag(Flag::Carry, true);
				} else {
					self.set_flag(Flag::Carry, false);
				}
			}
			/*Inst::DecBy16(location, amount) => {
				self.sub_from_data_u16(&location, amount);
			}
			Inst::IncBy16(location, amount) => {
				self.add_to_data_u16(&location, amount);
			}*/
			Inst::SetFlag(flag, on) => {
				self.set_flag(flag, on);
			}
			Inst::InvertFlag(flag) => {
				self.set_flag(flag, !self.get_flag(flag));
			}
			Inst::RepeatNextRegTimes{reg, until_zero_flag} => {
				// https://www.felixcloutier.com/x86/rep:repe:repz:repne:repnz
				// Note that 0xf3 could be a REPZ or just a REP depending on the following opcode.
				// (CMPS and SCAS are the only ones where the zero flag is checked, which are
				// 0xa6, 0xa7, 0xae, 0xaf).
				let repeat_inst = self.parse_instruction();
				
				let consider_zero_flag = match repeat_inst {
					// TODO: Add the CMPS and SCAS instructions when they are implemented.
					_ => false,
				};
				
				//dbg!(self.get_reg_u16(reg));
				while self.get_reg_u16(reg) != 0 {
					// TODO: Service pending interrupts
					self.apply_instruction(&repeat_inst, event_handler);
					self.sub_from_reg(reg, 1);
					//dbg!(self.get_reg_u16(reg));
					/*if self.get_reg_u16(reg) == 0 {
						break;
					}*/
					if consider_zero_flag {
						if let Some(until_zero_flag) = until_zero_flag {
							if self.get_flag(Flag::Zero) == until_zero_flag { 
								break;
							}
						}
					}
				}
			}
			Inst::MovReg32{source_h, source_l, dest_h, dest_l} => {
				let value_h = self.get_reg_u16(source_h);
				self.set_reg_u16(dest_h, value_h);
				let value_l = self.get_reg_u16(source_l);
				self.set_reg_u16(dest_l, value_l);
			}
			Inst::Call(offset) => {
				let old_ip = self.get_reg_u16(Reg::IP);
				self.push_u16(old_ip);
				// TODO: Maybe the offset should just be i16
				self.set_reg_u16(Reg::IP, (old_ip as i32 + offset) as u16);
			}
			Inst::Ret{extra_pop} => {
				let ip = self.pop_u16();
				self.set_reg_u16(Reg::IP, ip);
				self.add_to_reg(Reg::SP, extra_pop);
			}
			Inst::CallAbsolute{ip, cs} => {
				self.push_u16(self.get_reg_u16(Reg::CS));
				self.push_u16(self.get_reg_u16(Reg::IP));
				self.set_reg_u16(Reg::CS, cs);
				self.set_reg_u16(Reg::IP, ip);
			}
			Inst::CallAbsoluteWithAddress{seg, ref address16} => {
				let addr = self.get_seg_offset(seg, self.calculate_effective_address16(&address16));
				let ip = self.peek_u16(addr);
				let cs = self.peek_u16(addr + 2);
				self.push_u16(self.get_reg_u16(Reg::CS));
				self.push_u16(self.get_reg_u16(Reg::IP));
				self.set_reg_u16(Reg::CS, cs);
				self.set_reg_u16(Reg::IP, ip);
			}
			Inst::RetAbsolute{extra_pop} => {
				let ip = self.pop_u16();
				let cs = self.pop_u16();
				self.set_reg_u16(Reg::IP, ip);
				self.set_reg_u16(Reg::CS, cs);
				self.add_to_reg(Reg::SP, extra_pop);
			}
			Inst::Jump{ref condition, offset} => {
				let do_jump = if let Some(condition) = condition {
					self.test_jump_condition(condition)
				} else {
					true
				};
				
				if do_jump {
					//println!("----- JUMP -----");
					let ip = self.get_reg_u16(Reg::IP);
					self.set_reg_u16(Reg::IP, (ip as i32 + offset) as u16);
				}
			}
			Inst::JumpAbsolute{seg, ref address16} => {
				let addr = self.get_seg_offset(seg, self.calculate_effective_address16(address16));
				let ip = self.peek_u16(addr);
				let cs = self.peek_u16(addr + 2);
				self.set_reg_u16(Reg::IP, ip);
				self.set_reg_u16(Reg::CS, cs);
				//dbg!((cs, ip));
				/*for i in 0..16 {
					println!("{:?}: {:?}", addr + i, self.peek_u8(addr + i));
				}
				panic!();*/
			}
			Inst::JumpAndDecrementUntilZero{offset, dec_reg} => {
				self.sub_from_reg(dec_reg, 1);
				let count_value = self.get_reg_u16(dec_reg);
				if count_value != 0 {
					//println!("----- JUMP AND DEC ----- {:?}", count_value);
					
					let ip = self.get_reg_u16(Reg::IP);
					self.set_reg_u16(Reg::IP, (ip as i32 + offset) as u16);
				}
			}
			Inst::JumpZeroReg{offset, reg} => {
				let value = self.get_reg_u16(reg);
				if value == 0 {
					let ip = self.get_reg_u16(Reg::IP);
					self.set_reg_u16(Reg::IP, (ip as i32 + offset) as u16);
				}
			}
			Inst::Interrupt(interrupt_index) => {
				self.interrupt(interrupt_index);
			}
			Inst::InterruptIf(interrupt_index, flag) => {
				if self.get_flag(flag) {
					self.interrupt(interrupt_index);
				}
			}
			Inst::ReturnFromInterrupt => {
				let ip = self.pop_u16();
				let cs = self.pop_u16();
				let flags = self.pop_u16();
				self.set_reg_u16(Reg::IP, ip);
				self.set_reg_u16(Reg::CS, cs);
				self.set_reg_u16(Reg::Flags, flags);
			}
			Inst::Halt => {
				self.halted = true;
			}
			Inst::SignExtend8To16{ref source, ref destination} => {
				let value = self.get_data_u8(&source);
				let result = treat_i16_as_u16(treat_u8_as_i8(value) as i16);
				self.set_data_u16(&destination, result);
			}
			Inst::SignExtend16To32{ref source, ref destination_high, ref destination_low} => {
				let value = self.get_data_u16(&source);
				let result = treat_i32_as_u32(treat_u16_as_i16(value) as i32);
				self.set_data_u16(&destination_low, (result & 0xffff) as u16);
				self.set_data_u16(&destination_high, ((result >> 16) & 0xffff) as u16);
			}
			Inst::PortInput{ref port_index, destination: DataLocation::Size8(ref destination)} => {
				let port_index_value = self.get_data_u16(port_index);
				let port_result = event_handler.handle_port_input(self, port_index_value);
				self.set_data_u8(destination, (port_result & 0xff) as u8);
			}
			Inst::PortInput{ref port_index, destination: DataLocation::Size16(ref destination)} => {
				let port_index_value = self.get_data_u16(port_index);
				let port_result = event_handler.handle_port_input(self, port_index_value);
				self.set_data_u16(destination, port_result);
			}
			Inst::PortOutput{ref port_index, source: DataLocation::Size8(ref source)} => {
				let port_index_value = self.get_data_u16(port_index);
				let value = self.get_data_u8(source) as u16;
				event_handler.handle_port_output(self, port_index_value, value);
			}
			Inst::PortOutput{ref port_index, source: DataLocation::Size16(ref source)} => {
				let port_index_value = self.get_data_u16(port_index);
				let value = self.get_data_u16(source);
				event_handler.handle_port_output(self, port_index_value, value);
			}
		}
	}
	
	/// Parse and execute one instruction. If there are interrupts to be handled, this will do that instead.
	pub fn step(&mut self, event_handler: &mut EventHandler) {
		let non_maskable_bios_interrupt = 0x02;
		
		if self.get_flag(Flag::Interrupt)
				|| self.pending_interrupts.front() == Some(&non_maskable_bios_interrupt) {
			if let Some(interrupt_index) = self.pending_interrupts.pop_front() {
				self.interrupt(interrupt_index);
			}
		}
	
		// TODO: Improve this check.
		let ip = self.get_reg_u16(Reg::IP);
		let cs = self.get_reg_u16(Reg::CS);
		let mut handled_interrupt = false;
		//println!("{:x}, {:x}", ip, cs);
		if (ip & 0xff00) == 0x1100 && cs == 0xf000 {
			self.set_flag(Flag::Interrupt, true);
			let interrupt_index = (ip & 0xff) as u8;
			let interrupt_result = event_handler.handle_interrupt(self, interrupt_index);
			match interrupt_result {
				InterruptResult::Return => {
					self.return_from_interrupt();
				}
				InterruptResult::Stop => {
					panic!();
				}
			}
			
			handled_interrupt = true;
		}
		
		if !handled_interrupt {
			let inst = self.parse_instruction();
			if self.number_of_parsed_instructions > 2000000 {
				println!("MEM: {:?}", &self.memory[0xb8000..0xb8000+0x1000]);
				panic!();
			}
			//println!("{:?}", inst);
			self.apply_instruction(&inst, event_handler);
		}
	}
}
