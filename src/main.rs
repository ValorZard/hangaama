use hangaama::run;

fn main() {
    // await our future
    // dont use block_on inside of async function if we do WASM
    // futures have to be run using the browser's tools
    // otherwise it will crash
    pollster::block_on(run());
}
