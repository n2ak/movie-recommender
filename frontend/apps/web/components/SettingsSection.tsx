import { useAuthStore, UserInfo } from "@/hooks/useAuthStore";
import { ProfileSettingsFormState } from "@/lib/actions/FormStates";
import { changeProfileSettingsAction, logOut } from "@/lib/actions/user";
import { signOut, useSession } from "next-auth/react";
import { useActionState, useState } from "react";
import { ColStack } from "./Container";
import DeleteAccountModal from "./DeleteAccountModal";
import FormField from "./FormField";
import { useSnackBar } from "./providers/SnackBarProvider";
import { Button } from "./ui/button";

export default function SettingsSection({ user }: { user: UserInfo }) {
  const [saving, setSaving] = useState(false);
  const { update } = useSession();
  const snackbar = useSnackBar();

  const [state, formAction] = useActionState<
    ProfileSettingsFormState,
    FormData
  >(
    async (prevState, formData) => {
      const data = {
        username: (formData.get("name") as string) || "",
        email: prevState.data.email,
      };
      const res = await changeProfileSettingsAction(data);
      setSaving(false);
      if (res.message) {
        snackbar.warning("Error: " + res.message, 5000);
      } else {
        snackbar.success("Saved.", 1000);
        await update({
          name: data.username,
        });
      }
      return {
        data: {
          email: data.email,
          name: data.username,
        },
      };
    },
    {
      data: {
        name: user.username || "",
        email: user.email || "",
      },
    }
  );
  const [settings, setSettings] = useState(state.data);
  return (
    <>
      <h1 className="text-2xl font-semibold mb-4">Settings</h1>
      <ColStack className="gap-2 not-dark:bg-white p-4 rounded-lg shadow-sm">
        <form className="flex flex-col gap-2" action={formAction}>
          <div className="grid gap-6 grid-cols-2">
            <FormField
              name="name"
              type="text"
              placeholder="Username"
              label="Username"
              className="w-full"
              value={settings.name}
              error={state.errors?.name}
              onChange={(v) =>
                setSettings({
                  ...settings,
                  name: v,
                })
              }
            />
            <FormField
              name="email"
              type="email"
              placeholder="email"
              label="Email"
              disabled
              value={settings.email}
              error={state.errors?.email}
            />
          </div>
          <div>
            <Button
              className="!h-[40px] !w-[100px] float-right"
              disabled={saving}
              type="submit"
              onClick={() => setSaving(true)}
            >
              Save
            </Button>
          </div>
        </form>
        <DeleteAccount />
      </ColStack>
    </>
  );
}

function DeleteAccount() {
  const [open, setOpen] = useState(false);
  const { clearUser } = useAuthStore();
  return (
    <div>
      <Button
        onClick={() => setOpen(true)}
        className="dark:!bg-red-700 dark:hover:!bg-red-800"
      >
        Delete account.
      </Button>
      {open && (
        <DeleteAccountModal
          open={open}
          onCancel={() => setOpen(false)}
          onDelete={() => {
            setOpen(false);
            logOut();
            signOut();
            clearUser();
          }}
        />
      )}
    </div>
  );
}
