import { PrismaClient } from "@prisma/client";

async function connectToDatabase(client: PrismaClient) {
  try {
    await client.$connect();
    console.log('Successfully connected to the database.');
  } catch (error) {
    console.error('Prisma Client initialization error:', (error as any).message);
  }
}

const prismaClientSingleton = () => {
  const client = new PrismaClient();
  connectToDatabase(client);
  return client;
};

declare const globalThis: {
  prismaGlobal: ReturnType<typeof prismaClientSingleton>;
} & typeof global;

export const prismaClient = globalThis.prismaGlobal ?? prismaClientSingleton();

if (process.env.NODE_ENV !== "production")
  globalThis.prismaGlobal = prismaClient;
