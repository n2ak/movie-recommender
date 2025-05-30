"use server";
import { signIn, signOut } from "@/auth";
import { userDB } from "@repo/database";
import { AuthError } from "next-auth";
import { parseProfileSettings } from "../validation";
import { LoginFormState } from "./FormStates";
import { cachedQuery } from "./redisClient";
import { timedAction } from "./utils";

export const changeProfileSettingsAction = timedAction(
  "changeProfileSettingsAction",
  async (data: { username: string; email: string; userId: number }) => {
    parseProfileSettings(data);
    const newData = await userDB.changeProfileSettings(data);
    return {
      username: newData.username,
      email: data.email,
    };
  }
);

export async function authenticate(
  prevState: LoginFormState,
  formData: FormData
): Promise<LoginFormState> {
  try {
    await signIn("credentials", formData);
    return prevState;
  } catch (error) {
    if (error instanceof AuthError) {
      switch ((error as any).type) {
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

const getUserInfo = timedAction(
  "getUserInfo",
  cachedQuery(userDB.getUserInfo, ({ userId }) => `userInfo:${userId}`)
);
export { getUserInfo };
export const deleteAccount = timedAction(
  "deleteAccount",
  async ({ password, userId }: { password: string; userId: number }) => {
    const passwordMatch = await userDB.passwordMatch(userId, password);
    if (!passwordMatch) return false;
    return await userDB.deleteAccount(userId);
  }
);
