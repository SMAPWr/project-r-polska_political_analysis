import './App.css';
import Avatar from '@material-ui/core/Avatar';
import Button from '@material-ui/core/Button';
import CssBaseline from '@material-ui/core/CssBaseline';
import TextField from '@material-ui/core/TextField';
import LockOutlinedIcon from '@material-ui/icons/Twitter';
import Typography from '@material-ui/core/Typography';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import CircularProgress from '@material-ui/core/CircularProgress';
import LinearProgress from '@material-ui/core/LinearProgress'
import Grid from '@material-ui/core/Grid';
import Select from '@material-ui/core/Select';
import PoliticalPlot from './PoliticalPlot';
import { useState } from 'react';
import { predict_hashtag, predict_twitter_user } from './Model'

const useStyles = makeStyles((theme) => ({
  paper: {
    marginTop: theme.spacing(8),
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  avatar: {
    margin: theme.spacing(1),
    // backgroundColor: theme.palette.primary.main,
    backgroundColor: "#08a0e9"
  },
  form: {
    width: '100%', // Fix IE 11 issue.
    marginTop: theme.spacing(1),
  },
  submit: {
    margin: theme.spacing(3, 0, 2),
  },
  plot: {
    marginTop: -50
  },
  formControl: {
    margin: theme.spacing(1),
    marginTop: theme.spacing(10),
    minWidth: 120,
  },
  progress: {
    width: "100%",
    // marginTop: 50
  },
  plot: {
    marginTop: -50
  },
  title: {
    margin: 20, 
    marginBottom:45
  }
}));

function PredictOnTwitter() {
  const classes = useStyles();
  const [economic, setEconomic] = useState(null)
  const [worldview, setWorldview] = useState(null)
  const [text, setText] = useState("")
  const [byHashtag, setbyHashtag] = useState(true);
  const [loading, setLoading] = useState(false);
  const [tweets, setTweets] = useState([]);

  const handleButtonClick = async () => {
    setLoading(true)
    setEconomic(null)
    setWorldview(null)
    setTweets([])
    const data = byHashtag ? await predict_hashtag(text) : await predict_twitter_user(text)

    setEconomic(data.economic)
    setWorldview(data.worldview)
    setTweets(data.tweets)
    setLoading(false)
  }

  return (
    <Container component="main" maxWidth="md">
      <CssBaseline />
      <div className={classes.paper}>
        <Avatar className={classes.avatar}>
          <LockOutlinedIcon/>
        </Avatar>
        <Typography component="h1" variant="h5" className={classes.title}>
          Analize political attitude of tweets
        </Typography>

        <Grid container>
        
          <Grid item xs={6}>
            <FormControl className={classes.formControl} fullWidth>
              <InputLabel id="demo-simple-select-label">By</InputLabel>
              <Select
                labelId="demo-simple-select-label"
                id="demo-simple-select"
                value={byHashtag}
                onChange={(event) => { setbyHashtag(event.target.value) }}
              >
                <MenuItem value={true}>hashtag</MenuItem>
                <MenuItem value={false}>username</MenuItem>
              </Select>

            </FormControl>
            <TextField
              variant="outlined"
              margin="normal"
              required
              id="text"
              label={byHashtag ? "hashtag" : "username"}
              name="text"
              fullWidth
              value={text}
              onChange={(event => setText(event.target.value))}
              autoFocus
            />


            <Button
              type="submit"
              variant="contained"
              color="primary"
              fullWidth
              className={classes.submit}
              onClick={handleButtonClick}
            >
              Analize
        </Button>
        {loading ? <LinearProgress className={classes.progress}/> : null}
          </Grid>

          <Grid item xs={6} className={classes.plot}>
            <PoliticalPlot economic={economic} worldview={worldview} />
          </Grid>
        </Grid>

        
        


      </div>
    </Container>
  );
}

export default PredictOnTwitter;
