// "use server";
import authDb from "@repo/database/authDb";

import NextAuth, { User } from "next-auth";
import Credentials from "next-auth/providers/credentials";
import { parseCredentials } from "./_lib/validation";

export const checkPassword = async (
  email: string,
  password: string
): Promise<User | null> => {
  try {
    const user = await authDb.getByEmail(email);
    if (!!user) {
      // const passwordsMatch = await bcrypt.compare(password, user.password);
      const passwordsMatch = password === user.password;
      if (passwordsMatch)
        return {
          id: `${user.id}`,
          email: user.email,
          name: user.username,
        };
    }
  } catch (error) {
    console.log("Failed to fetch user " + error);
    // errorToJSON(error as any).message);
  }
  return null;
};

export const { signIn, signOut, auth, handlers } = NextAuth({
  providers: [
    Credentials({
      credentials: {
        username: { label: "Username" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        const { data } = parseCredentials(credentials);
        if (!!data) {
          const user = await checkPassword(data.email, data.password);
          return user;
        }
        return null;
      },
    }),
  ],
  pages: {
    signIn: "/auth/login",
  },
  callbacks: {
    authorized({ auth, request: { nextUrl, method } }) {
      // if (nextUrl.pathname.startsWith("/api/auth/session")) return true;
      const isLoggedIn = !!auth?.user;
      console.log("a message", isLoggedIn, nextUrl.pathname, method);
      const inLoginPage = nextUrl.pathname.startsWith("/auth/login");
      if (isLoggedIn && inLoginPage)
        return Response.redirect(new URL("/home", nextUrl));

      if (isLoggedIn) return true;
      if (inLoginPage) return true;
      return Response.redirect(new URL("/auth/login", nextUrl));
    },
    session: async ({ session, token, trigger }) => {
      session.user.email = token.email as string;
      session.user.id = token.id as string;
      session.user.name = token.name as string;
      // if (trigger === "update" && token.email) {
      //   console.log("Updating Session");
      //   const u = await authDb.getByEmail(token.email);
      //   if (u) {
      //     session.user.email = u.email;
      //     session.user.id = `${u.id}`;
      //     session.user.name = u.username;
      //   }
      // }
      return session;
    },
    jwt: async ({ token, user, trigger, session }) => {
      if (!!user) {
        token.email = user.email;
        token.id = user.id;
        token.name = user.name;
      }
      if (trigger === "update" && session.name) {
        console.log("Updating token");
        token.name = session.name;
      }
      return token;
    },
  },
});
