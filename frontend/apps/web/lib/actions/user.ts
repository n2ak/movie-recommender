"use server";
import { signOut } from "@/auth";
import { userDB } from "@repo/database";
import logger from "../logger";
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
const MAX_ACCOUNT_DELETION_ATTEMPS = 3;
export const deleteAccount = timedAction(
  "deleteAccount",
  async ({ password, userId }: { password: string; userId: number }) => {
    let ttl: number | undefined = 3;
    if (process.env.NODE_ENV !== "production") {
      ttl = undefined;
    }
    const counter = await incCachedCounter(
      `deleteAccountCounter:${userId}`,
      ttl
    );

    if (counter > MAX_ACCOUNT_DELETION_ATTEMPS) {
      logger.error(
        { counter, userId, max_attemps: MAX_ACCOUNT_DELETION_ATTEMPS },
        "Too many attempts to delete account"
      );
      throw new CustomError("Too much requests");
    }

    const passwordMatch = await userDB.passwordMatch(userId, password);
    if (!passwordMatch) return false;
    return await userDB.deleteAccount(userId);
  }
);
