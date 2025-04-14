export type ProfileSettingsFormState = FormState<{
  name: string;
  email: string;
}>;
export type LoginFormState = FormState<{
  username: string;
  password: string;
}>;
type FormState<T> = {
  data: T;
  errors?: { [key in keyof T]?: string };
  message?: string;
};
