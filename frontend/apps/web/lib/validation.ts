import { z } from "zod";
import { MAX_RATING } from "./constants";

const credentialsSchema = z.object({
  usernameOrEmail: z.union([z.string().email(), z.string()]),
  password: z.string().min(6),
});

const ratingSchema = z.object({
  rating: z
    .number({
      message: "Rating must be a number",
    })
    .min(1, `Rating must be in range [0,${MAX_RATING}]`)
    .max(MAX_RATING, `Rating must be in range [0,${MAX_RATING}]`),
  title: z.string().min(1).max(200),
  text: z.string().min(1).max(5000),
});
const profileSettingsSchema = z.object({
  name: z.string().min(4, "Username should be atleast 4 chars."),
});

export function parseCredentials<T>(obj: T) {
  return parse(obj, credentialsSchema);
}

export function parseRating<T>(obj: T) {
  return parse(obj, ratingSchema);
}

export function parseProfileSettings<T>(obj: T) {
  return parse(obj, profileSettingsSchema);
}

function parse<T, O>(obj: T, schema: z.Schema<O>) {
  const parsed = schema.safeParse(obj);
  if (!parsed.success) {
    throw new ValidationError(undefined, parsed.error.flatten().fieldErrors);
  }
  return parsed.data;
}

export class ValidationError extends Error {
  constructor(
    message: string | undefined,
    public errors: object
  ) {
    if (!message) {
      message = Object.values(errors)[0] as string;
    }
    super(message);
  }
}
