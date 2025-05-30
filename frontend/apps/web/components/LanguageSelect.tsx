import {
  languages,
  useDictionary,
  useLanguageStore,
} from "@/hooks/useLanguageStore";
import Select from "./Select";

export default function LanguageSelect() {
  const dict = useDictionary();
  const { language, setLanguage } = useLanguageStore();

  return (
    <Select
      defaultValue={language}
      label={dict.language}
      onValueChange={setLanguage as (s: string) => void}
      values={languages}
    />
  );
  // return (
  //   <Select.Root
  //     defaultValue={language}
  //     onValueChange={(v) => setLanguage(v as any)}
  //   >
  //     <Select.Trigger />
  //     <Select.Content>
  //       <Select.Group>
  //         <Select.Label>{dict.language}</Select.Label>
  //         {languages.map((l) => (
  //           <Select.Item key={l} value={l}>
  //             {l}
  //           </Select.Item>
  //         ))}
  //       </Select.Group>
  //     </Select.Content>
  //   </Select.Root>
  // );
}
