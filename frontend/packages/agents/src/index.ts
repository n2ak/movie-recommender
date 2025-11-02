import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { MemorySaver } from "@langchain/langgraph";
import { createAgent, HumanMessage } from "langchain";
import { tools } from "./tools";

export const SYSTEM_MESSAGE = `
You are helpfull chatbot, your task is to recommend movies based on user queries.
You can optionally ask the user to clarify the query or privide more information.
If an action helps, choose exactly ONE available tool that best achieves the outcome.
If no tool fits, answer with an apology for the lack of information.
Always wrap the movie name with a <id=#ID> and </id> tags with #ID being the movie ID.
Suggest movies in a bullet list.
`.trim();

const agent = createAgent({
    model: new ChatGoogleGenerativeAI({ model: "gemini-2.5-flash", apiKey: process.env.GOOGLE_API_KEY }),
    tools,
    checkpointer: new MemorySaver(),
    systemPrompt: SYSTEM_MESSAGE,
});

export const ask = async (msg: string, thread_id: string) => {
    const answer = await agent.invoke(
        { messages: [new HumanMessage(msg)] },
        { configurable: { thread_id } }
    );
    return answer;
}