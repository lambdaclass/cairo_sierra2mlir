use core::num::traits::WrappingSub;
use core::num::traits::WrappingAdd;
use core::{
    traits::Default, array::{ArrayTrait, SpanTrait}, debug::PrintTrait, dict::Felt252DictTrait,
    option::OptionTrait, traits::Into,
};

fn generate_jump_table(program: @Array<u8>) -> Felt252Dict<u32> {
    let mut jump_table: Felt252Dict<u32> = Default::default();
    let mut prev_table: Felt252Dict<u32> = Default::default();

    let mut pc = 0_u32;
    let mut frame = 0;
    loop {
        if pc >= program.len() {
            break ();
        }

        let op_code = *program[pc];
        if op_code == '[' {
            prev_table.insert(pc.into(), frame);
            frame = pc;
        } else if op_code == ']' {
            jump_table.insert(frame.into(), pc);
            jump_table.insert(pc.into(), frame);

            frame = prev_table.get(frame.into());
        }

        pc += 1;
    };

    jump_table
}

fn run_program(program: @Array<u8>, input: Option<Span<u8>>) {
    let mut memory: Felt252Dict<u8> = Default::default();
    let mut jump_table = generate_jump_table(program);

    let mut pc = 0;
    let mut mp = 0;

    loop {
        if pc >= program.len() {
            break ();
        }

        let op_code = *program[pc];
        if op_code == '>' {
            mp += 1;
        } else if op_code == '<' {
            mp -= 1;
        } else if op_code == '+' {
            memory.insert(mp, memory.get(mp).wrapping_add(1));
        } else if op_code == '-' {
            memory.insert(mp, memory.get(mp).wrapping_sub(1));
        } else if op_code == '.' {
            memory.get(mp).print();
        } else if op_code == ',' {
            memory
                .insert(
                    mp,
                    match input {
                        Option::Some(mut x) => match x.pop_front() {
                            Option::Some(x) => *x,
                            Option::None(_) => 0,
                        },
                        Option::None(_) => 0,
                    }
                );
        } else if op_code == '[' {
            if memory.get(mp) == 0 {
                pc = jump_table.get(pc.into());
            }
        } else if op_code == ']' {
            pc = jump_table.get(pc.into()) - 1;
        } else {
            let mut panic_msg = ArrayTrait::<felt252>::new();
            panic_msg.append('Invalid opcode.');
            panic_msg.append(op_code.into());
            panic(panic_msg);
        }

        pc += 1;
    }
}

fn main() {
    let mut program = ArrayTrait::<u8>::new();

    // Golfed hello world program (from Wikipedia).
    program.append('+');
    program.append('[');
    program.append('-');
    program.append('-');
    program.append('>');
    program.append('-');
    program.append('[');
    program.append('>');
    program.append('>');
    program.append('+');
    program.append('>');
    program.append('-');
    program.append('-');
    program.append('-');
    program.append('-');
    program.append('-');
    program.append('<');
    program.append('<');
    program.append(']');
    program.append('<');
    program.append('-');
    program.append('-');
    program.append('<');
    program.append('-');
    program.append('-');
    program.append('-');
    program.append(']');
    program.append('>');
    program.append('-');
    program.append('.');
    program.append('>');
    program.append('>');
    program.append('>');
    program.append('+');
    program.append('.');
    program.append('>');
    program.append('>');
    program.append('.');
    program.append('.');
    program.append('+');
    program.append('+');
    program.append('+');
    program.append('[');
    program.append('.');
    program.append('>');
    program.append(']');
    program.append('<');
    program.append('<');
    program.append('<');
    program.append('<');
    program.append('.');
    program.append('+');
    program.append('+');
    program.append('+');
    program.append('.');
    program.append('-');
    program.append('-');
    program.append('-');
    program.append('-');
    program.append('-');
    program.append('-');
    program.append('.');
    program.append('<');
    program.append('<');
    program.append('-');
    program.append('.');
    program.append('>');
    program.append('>');
    program.append('>');
    program.append('>');
    program.append('+');
    program.append('.');

    run_program(@program, Option::None(()));
}
