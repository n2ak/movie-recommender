"use server"
import { ask } from "@repo/agents";


export const invokeAgent = async ({ msg, conversation_id }: { msg: string, conversation_id: string }) => {
    // TODO: conversations sould be temp stored
    const resp = await ask(msg, conversation_id);
    return resp.messages[resp.messages.length - 1]?.content;
};

