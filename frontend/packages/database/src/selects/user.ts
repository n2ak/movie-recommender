import { Prisma } from "@prisma/client";

export const userSelect = Prisma.validator<Prisma.UserModelSelect>()({
  id: true,
  username: true,
  email: true,
  createdAt: true,
});
export const findUserByEmail = (email: string) =>
  Prisma.validator<Prisma.UserModelFindFirstArgs>()({
    where: { email, ...userWhere },
    select: userSelect,
  });
export const findUserById = (id: number) =>
  Prisma.validator<Prisma.UserModelFindFirstArgs>()({
    where: { id, ...userWhere },
    select: userSelect,
  });

export const userWhere = Prisma.validator<Prisma.UserModelWhereInput>()({
  // active: true,
});
