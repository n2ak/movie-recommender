"use server";
import { signOut } from "@/auth";
import { userDB } from "@repo/database";
import { parseProfileSettings } from "../validation";
import { cachedQuery, cachedCounter as incCachedCounter } from "./redisClient";
import { CustomError, timedAction } from "./utils";

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
    const counter = await incCachedCounter(`deleteAccountCounter:${userId}`);
    console.log("************", { counter });

    if (counter > 3) {
      throw new CustomError("Too much requests");
    }

    const passwordMatch = await userDB.passwordMatch(userId, password);
    if (!passwordMatch) return false;
    return await userDB.deleteAccount(userId);
  }
);
