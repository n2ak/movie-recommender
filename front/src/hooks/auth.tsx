import { createContext, useContext, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

interface UserContextType {
    user: any,
    logout: any,
    setUser: any

}
const UserContext = createContext<UserContextType>({
    user: undefined,
    logout: undefined,
    setUser: undefined
});

export function AuthProvider({ children }: any) {
    const [user, setUser] = useState<any>(null);
    const navigate = useNavigate();
    const logout = () => {
        //TODO
        console.log("Logging out");

        setUser(null);
        // navigate("/", { replace: true });
    };
    // const value = useMemo(
    //     () => ({
    //         user,
    //         logout,
    //         setUser
    //     }),
    //     [user]
    // );
    return <UserContext.Provider value={{
        user,
        logout,
        setUser
    }}>{children}</UserContext.Provider>
}

export function useAuth() {
    return useContext(UserContext);
}