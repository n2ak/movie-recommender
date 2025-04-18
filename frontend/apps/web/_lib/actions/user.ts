"use server";
import { signIn, signOut } from "@/auth";
import { userDB } from "@repo/database";
import { AuthError } from "next-auth";
import { parseProfileSettings, ValidationError } from "../validation";
import { LoginFormState, ProfileSettingsFormState } from "./FormStates";
import { actionWrapWithError, CustomError, getUserId } from "./utils";

export const changeProfileSettingsAction = actionWrapWithError()(async (
  data: ProfileSettingsFormState["data"]
) => {
  const { userId } = await getUserId();
  parseProfileSettings(data);
  try {
    const newData = await userDB.changeProfileSettings(userId, data);
    return {
      name: newData.username,
      email: data.email,
    };
  } catch {
    throw new CustomError("Couldn't change settings");
  }
});

export async function authenticate(
  prevState: LoginFormState,
  formData: FormData
): Promise<LoginFormState> {
  try {
    await signIn("credentials", formData);
    return prevState;
  } catch (error) {
    if (error instanceof AuthError) {
      switch (error.type) {
        case "CredentialsSignin":
          return {
            data: prevState.data,
            message: "Invalid credentials.",
          };
        default:
          return {
            data: prevState.data,
            message: "Something went wrong.",
          };
      }
    }
    throw error;
  }
  // redirect("/");
  // return {};
}

export async function logOut() {
  await signOut({
    redirectTo: "/auth/login",
  });
}

export const deleteAccount = actionWrapWithError<{
  password: string;
}>()(async (password: string) => {
  const { userId } = await getUserId();
  // TODO encrypt password
  const passwordMatch = await userDB.passwordMatch(userId, password);
  if (!passwordMatch) {
    throw new ValidationError(undefined, {
      password: "Invalid password",
    });
  }
  await new Promise((resolve) => setTimeout(resolve, 2000));
  await userDB.deleteAccount(userId);
});
