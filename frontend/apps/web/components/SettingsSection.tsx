import { changeProfileSettingsAction } from "@/_lib/actions/action";
import { ProfileSettingsFormState } from "@/_lib/actions/FormStates";
import { User } from "next-auth";
import { useSession } from "next-auth/react";
import { useActionState, useState } from "react";
import Button from "./Button";
import FormField from "./FormField";
import LanguageSelect from "./LanguageSelect";
import { useSnackBar } from "./providers/SnackBarProvider";

export default function SettingsSection({ user }: { user: User }) {
  const [saving, setSaving] = useState(false);
  const { update } = useSession();
  const snackbar = useSnackBar();
  // const s = useAuthStore(s=>s.User);
  const [state, formAction] = useActionState<
    ProfileSettingsFormState,
    FormData
  >(
    async (prevState, formData) => {
      const data = {
        name: (formData.get("name") as string) || "",
        email: prevState.data.email,
      };
      const res = await changeProfileSettingsAction(data);
      console.log({ res });
      setSaving(false);
      if (!!res.message) {
        snackbar.warning("Error: " + res.message, 5000);
      } else if (!!res.errors) {
        snackbar.warning("Errors: " + JSON.stringify(res.errors), 1000);
      } else {
        snackbar.success("Saved.", 1000);
        await update({
          name: data.name,
        });
        console.log("Updated?");
      }
      return res;
    },
    {
      data: {
        name: user.name || "",
        email: user.email || "",
      },
    }
  );
  const [settings, setSettings] = useState(state.data);

  return (
    <>
      <h1 className="text-2xl font-semibold mb-4">Settings</h1>
      <div className="bg-white p-4 rounded-lg shadow-sm">
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
              loading={saving}
              type="submit"
              onClick={() => setSaving(true)}
            >
              Save
            </Button>
          </div>
        </form>
        <LanguageSelect />

        {/* <div className="mt-4 space-y-2 text-sm text-gray-600">
          <div>
            <span className="font-medium">Location:</span> {user.location}
          </div>
          <div>
            <span className="font-medium">Website:</span>{" "}
            <a
              href={user.website}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              {user.website}
            </a>
          </div>
          <div>
            <span className="font-medium">Joined:</span> {user.joined}
          </div>
        </div> */}
      </div>
    </>
  );
}
