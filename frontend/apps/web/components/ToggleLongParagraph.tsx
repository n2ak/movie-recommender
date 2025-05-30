import { joinCN } from "@/lib/utils";
import { useState } from "react";

const ToggleLongParagraph = ({ text }: { text: string }) => {
  const [showP, setShowP] = useState(false);
  //TODO case when text is short.
  return (
    <div className="w-full">
      <div className={joinCN("my-3", showP ? "" : "line-clamp-2")}>{text}</div>
      <div className="text-right">
        <span
          className="text-xs underline cursor-pointer hover:text-blue-800"
          onClick={() => setShowP((s) => !s)}
        >
          {showP ? "show less" : "show more"}
        </span>
      </div>
    </div>
  );
};
export default ToggleLongParagraph;
