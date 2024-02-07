
import "./App.css";

import edgenLogo from "./assets/edgen_logo.svg"

function App() {

  return (
    <div className="container">
      <div>
        <img src={edgenLogo} className="logo edgen" alt="Edgen logo" />
      </div>

      <p style={{paddingTop: "2rem", fontSize:"2rem"}}>Edgen is now running, go to</p>

      <div style={{width: "100%", paddingTop: "2rem"}}>
        <button onClick={() => window.location.assign("https://chat.edgen.co")} style={{padding:"1.1rem", fontWeight:"bolder", fontSize:"1.5rem", backgroundColor:"#fdd000", color:"#333"}}>
          Edgen Chat
        </button>
      </div>

      <p style={{paddingTop: "2rem", fontSize:"2rem"}}>and start chatting!!</p>

    </div>
  );
}

export default App;
