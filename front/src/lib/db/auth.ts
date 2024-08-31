import "server-only";
import { prismaClient } from "./connect";
import { UserModel } from "@prisma/client";

const auth = {
  getByEmail: async function (email: string) {
    return await prismaClient.userModel.findFirst({
      where: { email },
      select: {
        id: true,
        username: true,
        email: true,
        password: true,
      },
    });
  },
};
export default auth;
