import { auth } from "@/auth";
import { PrismaClientInitializationError } from "@repo/database";
import { TIMING } from "../constants";
import logger, { startTimer } from "../logger";
import { ValidationError } from "../validation";

export class CustomError extends Error {
  constructor(message: string) {
    super(message);
  }
}

export async function getUserId() {
  const session = await auth();
  if (!session || !session.user || !session.user.id) {
    throw new CustomError("Unauthenticated");
  }
  return parseInt(session.user.id);
}

async function handleErrors<T>(promise: Promise<T>): Promise<{
  data?: T;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  message?: any;
}> {
  try {
    const data = await promise;
    return { data };
  } catch (e) {
    const ret = { message: "Unknown Error" };
    logger.error(e);
    if (e instanceof CustomError) {
      ret.message = e.message;
    } else if (e instanceof ValidationError) {
      ret.message = e.message;
    } else if (e instanceof PrismaClientInitializationError) {
      ret.message = e.message;
    }
    return ret;
  }
}

export function timedAction<T extends object, B, I = Omit<T, "userId">>(
  key: string,
  func: (a: T & { userId: number }) => Promise<B>
) {
  return async function (input: I) {
    const userId = await getUserId();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const inputs = { ...input, userId } as any;
    if (!TIMING) return await handleErrors(func(inputs));

    const stopTimer = startTimer();
    const data = await handleErrors(func(inputs));
    const duration = stopTimer();

    // TODO: long strings in the input
    logger.info(
      { input, key, duration: `${duration}ms`, successfull: !data.message },
      "Server Action"
    );

    return data;
  };
}
