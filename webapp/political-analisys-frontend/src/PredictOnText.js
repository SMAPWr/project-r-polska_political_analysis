import './App.css';
import Avatar from '@material-ui/core/Avatar';
import Button from '@material-ui/core/Button';
import CssBaseline from '@material-ui/core/CssBaseline';
import TextField from '@material-ui/core/TextField';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';
import Link from '@material-ui/core/Link';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';
import LockOutlinedIcon from '@material-ui/icons/ShortText';
import Typography from '@material-ui/core/Typography';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import PoliticalPlot from './PoliticalPlot';
import { useState } from 'react';
import { predict } from './Model'

const useStyles = makeStyles((theme) => ({
  paper: {
    marginTop: theme.spacing(8),
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  avatar: {
    margin: theme.spacing(1),
    backgroundColor: "#08a0e9",
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
  title: {
    margin: 20,
    marginBottom: 45
  }
}));

function PredictOnText() {
  const classes = useStyles();
  const [economic, setEconomic] = useState(0)
  const [worldview, setWorldview] = useState(0)
  const [text, setText] = useState("")

  const handleButtonClick = async () => {
    const data = await predict(text)
    setEconomic(data.economic)
    setWorldview(data.worldview)
  }

  return (
    <Container component="main" maxWidth="md">
      <CssBaseline />
      <div className={classes.paper}>
        <Avatar className={classes.avatar}>
          <LockOutlinedIcon />
        </Avatar>
        <Typography component="h1" variant="h5" className={classes.title}>
          Analyse political attitude of specified text
        </Typography>
        <Grid container>
          <Grid item xs={6}>
            <TextField
              variant="outlined"
              margin="normal"
              multiline
              rows={10}
              required
              fullWidth
              id="text"
              label="Text to analize"
              name="text"
              value={text}
              onChange={(event => setText(event.target.value))}

              autoFocus
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              color="primary"
              className={classes.submit}
              onClick={handleButtonClick}
            >
              Analize
            </Button>
          </Grid>
          <Grid item xs={6} className={classes.plot}>
            <PoliticalPlot economic={economic} worldview={worldview} />
          </Grid>
        </Grid>

      </div>
    </Container>
  );
}

export default PredictOnText;
