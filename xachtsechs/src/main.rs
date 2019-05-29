use std::cmp::Ordering;

mod bios_loader;
mod dos_event_handler;
mod dos_file_system;
mod exe_loader;
mod machine8086;
mod types;

use crate::dos_event_handler::{DosEventHandler, MachineType, PortStates};
use crate::dos_file_system::StandardDosFileSystem;
use crate::exe_loader::MzHeader;
use crate::machine8086::Machine8086;

// https://en.wikipedia.org/wiki/Program_Segment_Prefix
// https://toonormal.com/2018/06/07/notes-ms-dos-dev-for-intel-8086-cpus-using-a-modern-pc/
// - "DOS programs require that all programs start at the 256 byte boundary"
// https://www.daniweb.com/programming/software-development/threads/291076/whats-org-100h
// Super useful: http://www.mlsite.net/8086/
// https://en.wikibooks.org/wiki/X86_Assembly/Machine_Language_Conversion#Mod_/_Reg_/_R/M_tables
// https://www.felixcloutier.com/x86/rcl:rcr:rol:ror
// https://sites.google.com/site/microprocessorsbits/processor-control-instructions/lock

fn main() {
    let mut file = std::fs::File::open("ZZT.EXE").unwrap();
    let exe_header = MzHeader::parse(&mut file).unwrap();
    println!("{:#?}", exe_header);
    let mut machine = Machine8086::new(1024*1024*1);
    exe_header.load_into_machine(&mut machine, &mut file);
    let mut event_handler = DosEventHandler {
		machine_type: MachineType::EGA,
		video_mode: MachineType::EGA.lookup_video_mode(3).unwrap(),
		port_states: PortStates::new(),
		file_system: Box::new(StandardDosFileSystem::new()),
		seconds_since_start: 0.,
    };
    event_handler.init_machine(&mut machine);
    let mut step_count = 0;
    loop {
		if step_count % 1000 == 0 {
			//println!("interrupt 0x08");
			machine.interrupt_on_next_step(0x08);
			event_handler.seconds_since_start += 0.01;
		}
		if step_count % 100000 == 0 {
			event_handler.set_cga_vertial_retrace(true);
		}
		//println!("before");
		machine.step(&mut event_handler);
		//println!("after");
		step_count += 1;
	}
}
