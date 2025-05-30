import { userDB } from "@repo/database";
import NextAuth from "next-auth";
import Credentials from "next-auth/providers/credentials";
import { parseCredentials } from "./lib/validation";

const { signOut, auth, handlers } = NextAuth({
  providers: [
    Credentials({
      credentials: {
        usernameOrEmail: { label: "Username or email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(cred) {
        console.log(cred);

        const parsedCred = parseCredentials(cred);
        const user = await userDB.passwordMatchByUserNameOrEmail(
          parsedCred.usernameOrEmail,
          parsedCred.password
        );
        if (!user) {
          throw new Error("Invalid credentials");
        }
        const ret = {
          id: `${user.id}`,
          email: user.email,
          name: user.username,
        };
        console.log(ret, "User logged in");
        return ret;
      },
    }),
  ],
  pages: {
    signIn: "/auth/login",
  },
  callbacks: {
    authorized({ auth, request: { nextUrl } }) {
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
export { auth, handlers, signOut };
