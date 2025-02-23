use core::circuit::{
    RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add,
    circuit_sub, circuit_mul, circuit_inverse, EvalCircuitTrait, u384,
    CircuitOutputsTrait, CircuitModulus, AddInputResultTrait, CircuitInputs,
};

fn main() -> u384 {
    let in1 = CircuitElement::<CircuitInput<0>> {};
    let in2 = CircuitElement::<CircuitInput<1>> {};
    let add = circuit_add(in1, in2);
    let mul = circuit_mul(add, in2);
    let inv1 = circuit_inverse(mul);
    let sub1 = circuit_sub(inv1, in1);

    let modulus = TryInto::<_, CircuitModulus>::try_into([17, 14, 14, 14]).unwrap();

    let outputs = (sub1,)
        .new_inputs()
        .next([9, 2, 9, 3])
        .next([5, 7, 0, 8])
        .done()
        .eval(modulus)
        .unwrap();

    outputs.get_output(sub1)
}
