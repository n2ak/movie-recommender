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

export async function passwordMatchByEmail(email: string, password: string) {
  return !!(await prismaClient.userModel.findFirst({
    where: { email, ...userWhere, password },
  }));
}

export async function changeProfileSettings(
  userId: number,
  data: {
    name: string;
  }
) {
  return await prismaClient.userModel.update({
    where: { id: userId, ...userWhere },
    data: {
      username: data.name,
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
