import { prismaClient } from "./connect";
import {
  findUserByEmail,
  findUserById,
  userSelect,
  userWhere,
} from "./selects/user";

export type User = NonNullable<Awaited<ReturnType<typeof getByEmail>>>;

export const getByEmail = async (email: string) =>
  await prismaClient.userModel.findFirst(findUserByEmail(email));

export const getById = async (id: number): Promise<User | null> =>
  await prismaClient.userModel.findFirst(findUserById(id));

export async function passwordMatch(id: number, password: string) {
  return (
    (
      await prismaClient.userModel.findFirst({
        where: { id, ...userWhere },
      })
    )?.password === password
  );
}

// getUserInfo function
export async function getUserInfo({ userId }: { userId: number }) {
  return await prismaClient.userModel.findFirst({
    where: { id: userId, ...userWhere },
    select: userSelect,
  });
}

export async function passwordMatchByUserNameOrEmail(
  usernameOrEmail: string,
  encryptedPassword: string
) {
  // TODO:
  return await prismaClient.userModel.findFirst({
    where: {
      OR: [
        {
          username: usernameOrEmail,
        },
        {
          email: usernameOrEmail,
        },
      ],
      password: encryptedPassword,
      ...userWhere,
    },
    select: userSelect,
  });
}

export async function changeProfileSettings(params: {
  userId: number;
  username: string;
  // email:string TODO
}) {
  return await prismaClient.userModel.update({
    where: { id: params.userId, ...userWhere },
    data: {
      username: params.username,
    },
    select: userSelect,
  });
}
export async function deleteAccount(userId: number) {
  return (
    (
      await prismaClient.userModel.update({
        where: {
          id: userId,
        },
        data: {
          active: false,
        },
      })
    ).active === false
  );
}
