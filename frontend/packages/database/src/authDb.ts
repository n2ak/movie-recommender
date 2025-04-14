import { prismaClient } from "./connect";

const authDb = {
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
  changeProfileSettings: async function (
    userId: number,
    data: {
      name: string;
    }
  ) {
    return await prismaClient.userModel.update({
      where: { id: userId },
      data: {
        username: data.name,
      },
      select: {
        username: true,
      },
    });
  },
};
export default authDb;
