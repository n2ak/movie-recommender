"use client";
import { useAuthStore } from "@/hooks/useAuthStore";
import Link from "next/link";
import * as React from "react";
// import { useFormState, useActionState } from "react-dom";
import { useDictionary } from "@/hooks/useLanguageStore";
import { signIn } from "next-auth/react";
import Button from "./Button";
import { ColStack } from "./Container";
import FormField from "./FormField";

export default function SignIn({ defaultPassword, defaultEmail }: any) {
  const dict = useDictionary();
  const { user, loading } = useAuthStore();
  // const [state, formAction] = React.useActionState<LoginFormState, FormData>(
  //   authenticate,
  //   {
  //     data: { username: defaultEmail, password: defaultPassword },
  //   }
  // );
  const [loginData, setLoginData] = React.useState<{
    username: string;
    password: string;
    message?: string;
  }>({
    username: defaultEmail,
    password: defaultPassword,
  });
  if (loading) {
    return null;
  }
  // if (user) {
  //   router.push("/home");
  //   return null;
  // }
  return (
    <>
      <div className="flex justify-center text-black">
        <div className="rounded-md bg-white min-w-[400px] px-10 py-5 shadow-2xl mb-3 dark:bg-3">
          <h2 className="font-bold text-5xl w-full flex m-auto mb-10">
            <div className="mx-auto">{dict.signIn}</div>
          </h2>
          <form
            noValidate
            onSubmit={async (e) => {
              e.preventDefault();
              try {
                await signIn("credentials", {
                  email: defaultEmail,
                  password: defaultPassword,
                  redirectTo: "/home",
                });
                setLoginData({
                  ...loginData,
                  message: undefined,
                });
              } catch (e: any) {
                console.error(e);
                setLoginData({
                  ...loginData,
                  message: e.message,
                });
              }
            }}
            className="flex flex-col w-full gap-2"
          >
            <ColStack className="gap-4">
              <FormField
                type="email"
                name="email"
                placeholder="your@email.com"
                value={loginData.username}
                onChange={(v) => {
                  setLoginData({
                    ...loginData,
                    username: v,
                  });
                }}
                hasError={!!loginData.message}
              />
              <FormField
                type="password"
                name="password"
                value={loginData.password}
                onChange={(v) => {
                  setLoginData({
                    ...loginData,
                    password: v,
                  });
                }}
                hasError={!!loginData.message}
                addPasswordTogle
              />
              {/* <FormControlLabel
              control={<Checkbox value="remember" color="primary" />}
              label="Remember me"
            /> */}
              {/* <ForgotPassword open={open} handleClose={handleClose} /> */}
              <Button type="submit" className="text-white">
                Sign In
              </Button>
              <div className="text-center">
                {"Don't have an account? "}
                <span>
                  <Link
                    href="/auth/register"
                    className="underline text-blue-900"
                  >
                    Sign up
                  </Link>
                </span>
              </div>
            </ColStack>
            {/* <input type="hidden" name="redirectTo" value="/home" /> */}
          </form>
          <div
            className="flex h-8 items-end space-x-1"
            aria-live="polite"
            aria-atomic="true"
          >
            {loginData.message}
          </div>
        </div>
      </div>
    </>
  );
}
