import readline from "readline";
import { ask } from "./index";
import { similaritySearch } from "./vectore_store";

function getInput(question: string): Promise<string> {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    return new Promise((resolve) =>
        rl.question(question, (answer) => {
            rl.close();
            resolve(answer);
        })
    );
}

async function chat() {
    while (true) {
        const input = await getInput("You: ");
        if (input.toLowerCase() === "quit") {
            break;
        }
        const resp = await ask(input, "0").then(r => {
            return r.messages[r.messages.length - 1]?.content;
        });
        console.log("Agent:", resp)
    }
}
async function test() {
    const results = await similaritySearch("a movie", 3);
    console.log(results);
}

chat();
// test()