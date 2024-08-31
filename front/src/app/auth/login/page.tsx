"use server";

import SignIn from "@/components/login";

export default async function Log() {
  return (
    <SignIn defaultPassword={"password"} defaultEmail={"user1@gmail.com"} />
  );
}
