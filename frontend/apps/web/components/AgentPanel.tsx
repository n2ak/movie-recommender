"use client";

import { useAuthStore } from "@/hooks/useAuthStore";
import { invokeAgent } from "@/lib/actions/agent";
import { ArrowDown, MessageSquare } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

interface Message {
  id: string;
  text: string;
  sender: "user" | "agent";
}

export function AgentPanel() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([{
    sender: "agent",
    id: uuidv4(),
    text: "Hi, what would you like to watch?"
  }]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState(uuidv4());
  const [pending, setPending] = useState(false);

  const inputEmptyOrPending = input.trim() === "" || pending;
  const handleSendMessage = async () => {
    if (inputEmptyOrPending) return;
    setPending(true);

    const userMessage: Message = {
      id: uuidv4(),
      text: input,
      sender: "user",
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    const agentResponse = await invokeAgent({ msg: input, conversation_id: conversationId });

    if (agentResponse) {
      const agentMessage: Message = {
        id: uuidv4(),
        text: `${agentResponse}`,
        sender: "agent",
      };
      setMessages((prev) => [...prev, agentMessage]);
      setPending(false);
    }
  };
  const user = useAuthStore((s) => s.user);
  if (!user) {
    return null;
  }

  return (
    <div className="fixed bottom-5 right-5 z-50">
      {isOpen ? (
        <div className="flex flex-col h-[600px] w-[400px] border rounded-lg bg-white dark:bg-black shadow-lg">
          <div className="flex justify-between items-center p-4 border-b">
            <h3 className="font-bold">Agent</h3>
            <div className="flex gap-2">
              <Button variant="ghost" size="sm" onClick={() => {
                setMessages([{
                  sender: "agent",
                  id: uuidv4(),
                  text: "Hi, what would you like to watch?"
                }]);
                setConversationId(uuidv4());
              }}>
                Clear
              </Button>
              <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)}>
                <ArrowDown />
              </Button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            {messages.map(msg =>
              <div key={msg.id}>
                {msg.sender === "user" ? <UserMessage msg={msg} /> : <AgentMessage msg={msg} />}
              </div>
            )}
          </div>
          <div className="p-4 border-t">
            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
              />
              <Button
                onClick={handleSendMessage}
                disabled={inputEmptyOrPending}
                className={inputEmptyOrPending ? "cursor-not-allowed" : ""}
              >
                Send
              </Button>
            </div>
          </div>
        </div>
      ) : (
        <Button onClick={() => setIsOpen(true)} size="lg" className="rounded-full w-16 h-16">
          <MessageSquare size={32} />
        </Button>
      )}
    </div>
  );
}

const UserMessage = ({ msg }: { msg: Message }) => {
  return <div
    className="flex justify-end"
  >
    <div
      className="max-w-xs rounded-lg px-4 py-2 my-1 bg-blue-500 text-white"
    >
      {msg.text}
    </div>
  </div>
}
const AgentMessage = ({ msg }: { msg: Message }) => {
  const elements = msg.text.split(/(<id=\d+>.*?<\/id>)/g).map((part, i) => {
    const match = part.match(/<id=(\d+)>(.*?)<\/id>/);

    if (match) {
      const [, id, name] = match;
      return (
        <Link key={i} href={`/movie/${id}`} className="text-blue-500 underline">
          {name}
        </Link>
      );
    }

    return <span key={i}>{part}</span>;
  });
  return <div
    className="flex justify-start"
  >
    <div
      className="max-w-xs rounded-lg px-4 py-2 my-1 bg-gray-200 text-black"
    >
      {elements}
    </div>
  </div>
}
