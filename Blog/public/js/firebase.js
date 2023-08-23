let firebaseConfig = {
    apiKey: "AIzaSyCF60NNnFk6w6RH3imFK8hyFWgjhHKywtY",
    authDomain: "blogging-website-29bc5.firebaseapp.com",
    projectId: "blogging-website-29bc5",
    storageBucket: "blogging-website-29bc5.appspot.com",
    messagingSenderId: "334073086611",
    appId: "1:334073086611:web:8bcbb5d6c6701ecfb3c366"
  };

  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);

  db = firebase.firestore();