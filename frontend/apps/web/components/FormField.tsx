export default function FormField({
  placeholder,
  className,
  type,
  value,
  onChange,
  error,
  name,
  label,
  disabled,
}: {
  placeholder?: string;
  value?: string;
  className?: string;
  onChange?: (v: string) => void;
  type: "password" | "email" | "text";
  error?: string;
  name?: string;
  label?: string;
  disabled?: boolean;
}) {
  return (
    <div className="">
      {label && (
        <label
          htmlFor={name}
          className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
        >
          {label}
        </label>
      )}
      <input
        disabled={!!disabled}
        value={value}
        name={name}
        onChange={(e) => {
          if (!!onChange) onChange(e.target.value);
        }}
        className={
          className +
          ` w-full flex-4/6 border border-black/30 focus:border-blue-500 rounded-md py-1.5 px-2 ${!!error ? "!border-red-600" : ""}`
        }
        type={type}
        placeholder={placeholder}
      />

      {!!error && (
        <span className="text-red-600 font-medium text-sm">{error}</span>
      )}
    </div>
  );
}
