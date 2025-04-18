import { userDB } from "@repo/database";

import NextAuth, { User } from "next-auth";
import Credentials from "next-auth/providers/credentials";
import { parseCredentials } from "./_lib/validation";

export const checkPassword = async (
  email: string,
  password: string
): Promise<User | null> => {
  try {
    //TODO could be one request to db
    // const passwordsMatch = await bcrypt.compare(password, user.password);
    if (await userDB.passwordMatchByEmail(email, password)) {
      const user = await userDB.getByEmail(email);
      if (user)
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
        if (data) {
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
      const inLoginPage = nextUrl.pathname.startsWith("/auth/login");
      if (isLoggedIn && inLoginPage)
        return Response.redirect(new URL("/home", nextUrl));

      if (isLoggedIn) return true;
      if (inLoginPage) return true;
      return Response.redirect(new URL("/auth/login", nextUrl));
    },
    session: async ({ session, token }) => {
      session.user.email = token.email as string;
      session.user.id = token.id as string;
      session.user.name = token.name as string;
      return session;
    },
    jwt: async ({ token, user, trigger, session }) => {
      if (user) {
        token.email = user.email;
        token.id = user.id;
        token.name = user.name;
      }
      if (trigger === "update" && session.name) {
        token.name = session.name;
      }
      return token;
    },
  },
});
