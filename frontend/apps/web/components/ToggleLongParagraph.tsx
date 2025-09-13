import { joinCN } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";

const ToggleLongParagraph = ({ text }: { text: string }) => {
  const [showP, setShowP] = useState(false);
  const [isOverflowing, setIsOverflowing] = useState(true);
  const textRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (textRef.current)
      setIsOverflowing(
        textRef.current.scrollHeight > textRef.current.clientHeight
      );
  }, [text]);

  return (
    <div className="w-full">
      <div
        className={joinCN("my-3", showP ? "" : "line-clamp-2")}
        ref={textRef}
      >
        {text}
      </div>
      {isOverflowing && (
        <div className="text-right">
          <span
            className="text-xs underline cursor-pointer hover:font-bold hover:text-primary"
            onClick={() => setShowP((s) => !s)}
          >
            {showP ? "show less" : "show more"}
          </span>
        </div>
      )}
    </div>
  );
};
export default ToggleLongParagraph;
