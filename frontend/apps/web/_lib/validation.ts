import { z } from "zod";

const credentialsSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
});

const ratingSchema = z.number().min(1).max(5);
const profileSettingsSchema = z.object({
  name: z
    .string({
      description: "keeek",
    })
    .min(4, "Username should be atleast 4 chars."),
});

export function parseCredentials(obj: any) {
  return parse(obj, credentialsSchema);
}

export function parseRating(obj: any) {
  return parse(obj, ratingSchema);
}
export function parseProfileSettings(obj: any) {
  return parse(obj, profileSettingsSchema);
}
function getErrors<I, O>(parsed: z.SafeParseReturnType<I, O>) {
  const sep = "\n";
  const errors: {
    [key in keyof I]?: string;
  } = {};
  parsed.error?.errors.forEach((err) => {
    err.path.forEach((p: any) => {
      if (!Object.hasOwn(errors, p)) {
        errors[p as keyof I] = "";
      }
      errors[p as keyof I] =
        (errors[p as keyof I] || "") + `${sep}${err.message}`;
    });
  });
  return errors;
}
function parse<O>(obj: any, schema: z.Schema<O>) {
  const parsed = schema.safeParse(obj);
  if (!parsed.success) {
    return {
      errors: getErrors(parsed),
    };
  }
  return {
    data: parsed.data,
  };
}
