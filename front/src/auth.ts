import NextAuth from "next-auth";
import { authConfig } from "./auth.config";
import { z } from "zod";
import type { User } from "@/lib/definitions";
import authDB from "@/lib/db/auth";
import bcrypt from "bcrypt";

import Credentials from "next-auth/providers/credentials";

export const getUser = async (email: string): Promise<User | null> => {
  try {
    const user = await authDB.getByEmail(email);
    return user as any;
  } catch (error) {
    // console.error(typeof error);
    console.log("Failed to fetch user " + error); // errorToJSON(error as any).message);
    return null;
  }
};
export const { signIn, signOut, auth, handlers } = NextAuth({
  ...authConfig,
  providers: [
    Credentials({
      credentials: {
        username: { label: "Username" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        const parsedCredentials = z
          .object({ email: z.string().email(), password: z.string().min(6) })
          .safeParse(credentials);
        console.log("Yoooooo");
        if (parsedCredentials.success) {
          const { email, password } = parsedCredentials.data;
          const user = await getUser(email);
          if (!user) return null;
          // const passwordsMatch = await bcrypt.compare(password, user.password);
          const passwordsMatch = password === user.password;

          console.log("passwordsMatch", passwordsMatch);
          if (passwordsMatch) return user as any;
        }
        // console.log("Invalid credentials", parsedCredentials.error?.issues);
        return null;
      },
    }),
  ],
});
