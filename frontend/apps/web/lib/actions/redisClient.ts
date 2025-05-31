import { Redis } from "ioredis";
import { CACHING, TTL } from "../constants";
import logger from "../logger";

const create = () => {
  const options = {
    host: process.env.REDIS_HOST,
    port: Number(process.env.REDIS_PORT),
    password: process.env.REDIS_PASSWORD,
  };
  logger.debug(options, "Redis options:");
  const redis = new Redis(options);
  return redis;
};

export function cachedQuery<I, O>(
  fetch: (i: I) => Promise<O>,
  getKey: (i: I) => string
): (i: I) => Promise<O> {
  return async function (a: I) {
    if (!CACHING) return await fetch(a);
    const key = getKey(a);
    const cached = await redisClient.get(key);
    if (cached) {
      logger.debug({ key }, "*****Cache hit");
      return JSON.parse(cached);
    }
    const data = await fetch(a);
    logger.debug({ key }, "*****Cache miss");
    if (data) redisClient.set(key, JSON.stringify(data), "EX", TTL);
    return data;
  };
}

export async function clearCacheKey(key: string, reason: string) {
  logger.debug({ key, reason }, "*****Cache clear");
  return await redisClient.del(key);
}

declare const globalThis: {
  redisClient: ReturnType<typeof create>;
} & typeof global;
const redisClient = globalThis.redisClient ?? create();
if (process.env.NODE_ENV !== "production") globalThis.redisClient = redisClient;

export async function cachedCounter(key: string) {
  const cache = await redisClient.get(key);

  let counter = 0;
  if (cache) {
    counter = parseInt(cache);
  }
  counter++;
  await redisClient.set(key, `${counter}`);
  return counter;
}
