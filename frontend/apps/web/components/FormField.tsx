import { EyeIcon, EyeOffIcon } from "lucide-react";
import { useState } from "react";
import { Input } from "./ui/input";

export default function Password({
  onChange,
  value,
  addToggle,
  name,
}: {
  value: string;
  onChange: (s: string) => void;
  addToggle?: boolean;
  name?: string;
}) {
  const [hidden, setHidden] = useState(true);
  const Eye = hidden ? EyeOffIcon : EyeIcon;
  return (
    <div className="relative">
      <Input
        type={hidden ? "password" : "text"}
        placeholder="Password"
        value={value}
        onChange={(v) => onChange(v.target.value)}
        name={name || "password"}
        required
      />
      {addToggle && (
        <Eye
          className="absolute top-1/2 transform -translate-y-1/2 right-0 mr-2 px-1 w-7 py-0 rounded-sm cursor-pointer hover:bg-primary dark:hover:text-black"
          onClick={() => setHidden(!hidden)}
        />
      )}
    </div>
  );
}
