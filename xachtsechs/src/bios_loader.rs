use crate::machine8086::Machine8086;
use crate::types::{DataLocation8, DataLocation16};

pub const BIOS_START: u32 = 0x40 << 4;
const fn bios_off_u8(offset: u32) -> DataLocation8 {
	DataLocation8::MemoryAbs(BIOS_START + offset)
}
const fn bios_off_u16(offset: u32) -> DataLocation16 {
	DataLocation16::MemoryAbs(BIOS_START + offset)
}

pub const BIOS_EQUIPMENT: DataLocation16 = bios_off_u16(10);
pub const BIOS_MEMORY_SIZE_KB: DataLocation16 = bios_off_u16(0x13);
pub const BIOS_VIDEO_MODE_INDEX: DataLocation8 = bios_off_u8(0x49);
pub const BIOS_TEXT_COLUMN_COUNT: DataLocation16 = bios_off_u16(0x4a);
pub const BIOS_TEXT_PAGE_BYTES: DataLocation16 = bios_off_u16(0x4c);
pub const BIOS_CURSOR_POSITION: [DataLocation16; 8] = [
	bios_off_u16(0x50), bios_off_u16(0x52), bios_off_u16(0x54), bios_off_u16(0x56),
	bios_off_u16(0x58), bios_off_u16(0x5a), bios_off_u16(0x5c), bios_off_u16(0x5e),
];
pub const BIOS_ACTIVE_VIDEO_PAGE: DataLocation8 = bios_off_u8(0x62);
pub const BIOS_VIDEO_IO_PORT_ADDRESS: DataLocation16 = bios_off_u16(0x63);
pub const BIOS_SYSTEM_TIMER_COUNTER_ADDR_U32: u32 = BIOS_START + 0x6c;
pub const BIOS_SYSTEM_TIMER_COUNTER_LOW: DataLocation16 = bios_off_u16(0x6c);
pub const BIOS_SYSTEM_TIMER_COUNTER_HIGH: DataLocation16 = bios_off_u16(0x6e);
pub const BIOS_TEXT_ROW_COUNT: DataLocation16 = bios_off_u16(0x84);
pub const BIOS_CHAR_HEIGHT: DataLocation16 = bios_off_u16(0x85);

// http://www.bioscentral.com/misc/bda.htm
pub fn initialise_bios_data_area(machine: &mut Machine8086) {
	// The BIOS Data Area starts at the start of the 0x40 segment.
	// Equipment
	machine.set_data_u16(&BIOS_EQUIPMENT, 0x0061);
	// Memory size in KB
	machine.set_data_u16(&BIOS_MEMORY_SIZE_KB, 640);
	// Text column count for the video mode
	machine.set_data_u16(&BIOS_TEXT_COLUMN_COUNT, 80);
	// Port for video I/O
	machine.set_data_u16(&BIOS_VIDEO_IO_PORT_ADDRESS, 0xd403);
}
