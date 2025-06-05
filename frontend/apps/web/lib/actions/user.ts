"use server";
import { userDB } from "@repo/database";
import logger from "../logger";
import { parseProfileSettings } from "../validation";
import {
  cachedQuery,
  clearCacheKey,
  cachedCounter as incCachedCounter,
} from "./redisClient";
import { CustomError, action } from "./utils";

export const changeProfileSettingsAction = action(
  "changeProfileSettingsAction",
  async (data: { username: string; userId: number }) => {
    parseProfileSettings(data);
    const newData = await userDB.changeProfileSettings(data);
    await clearCacheKey(`userInfo:${data.userId}`, "ProfileSettingsChange");
    return {
      username: newData.username,
    };
  }
);

const getUserInfo = action(
  "getUserInfo",
  cachedQuery(userDB.getUserInfo, ({ userId }) => `userInfo:${userId}`)
);
export { getUserInfo };
const MAX_ACCOUNT_DELETION_ATTEMPTS = 3;
export const deleteAccount = action(
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

    if (counter > MAX_ACCOUNT_DELETION_ATTEMPTS) {
      logger.error(
        { counter, userId, max_attemps: MAX_ACCOUNT_DELETION_ATTEMPTS },
        "Too many attempts to delete account"
      );
      throw new CustomError("Too much requests");
    }

    const passwordMatch = await userDB.passwordMatch(userId, password);
    if (!passwordMatch) return false;
    return await userDB.deleteAccount(userId);
  }
);
