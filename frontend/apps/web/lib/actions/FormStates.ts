export type ProfileSettingsFormState = FormState<{
  name: string;
  email: string;
}>;

type FormState<T> = {
  data: T;
  errors?: { [key in keyof T]?: string };
  message?: string;
};
