import { z } from "zod";

const credentialsSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
});

const ratingSchema = z
  .number({
    message: "Rating must be a number",
  })
  .min(1, "Rating must be in range [0,5]")
  .max(5, "Rating must be in range [0,5]");
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

function getErrors<I, O>(parsed: z.SafeParseReturnType<I, O>, key?: string) {
  const sep = "\n";
  const errors: {
    [key in keyof I]?: string;
  } = {};
  function join(p: keyof I, message: string) {
    errors[p] = (errors[p] ? `${errors[p]}${sep}` : "") + message;
  }
  parsed.error?.errors.forEach((err) => {
    if (err.path.length == 0 && !!key) {
      join(key as keyof I, err.message);
    } else {
      err.path.forEach((p: any) => {
        join(p, err.message);
      });
    }
  });
  return errors;
}

function parse<O>(obj: any, schema: z.Schema<O>) {
  const parsed = schema.safeParse(obj);
  if (!parsed.success) {
    throw new ValidationError(undefined, getErrors(parsed, "rating"));
  }
  return {
    data: parsed.data,
  };
}

export class ValidationError extends Error {
  constructor(
    message: string | undefined,
    public errors: any
  ) {
    if (!message) {
      message = Object.values(errors)[0] as string;
    }
    super(message);
  }
}
