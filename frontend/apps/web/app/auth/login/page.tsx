"use server";

import SignIn from "@/components/Login";

export default async function Log() {
  return (
    <SignIn defaultPassword={"password"} defaultEmail={"user1@gmail.com"} />
  );
}
