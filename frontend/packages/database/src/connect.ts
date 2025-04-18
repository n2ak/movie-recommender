import { PrismaClient } from "@prisma/client";

import "server-only";

const prismaClientSingleton = () => {
  const client = new PrismaClient();
  // .$extends({
  //   name: "logging",
  //   query: {
  //     $allModels: {
  //       async $allOperations({ model, operation, args, query }) {
  //         console.log(`[PRISMA] ${model}.${operation}`);
  //         const result = await query(args);
  //         return result;
  //       },
  //     },
  //   },
  // });
  return client;
};

declare const globalThis: {
  prismaGlobal: ReturnType<typeof prismaClientSingleton>;
} & typeof global;

export const prismaClient = globalThis.prismaGlobal ?? prismaClientSingleton();

if (process.env.NODE_ENV !== "production")
  globalThis.prismaGlobal = prismaClient;
