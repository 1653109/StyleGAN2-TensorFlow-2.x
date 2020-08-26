import React from 'react'
import { 
  Card,
  CardActionArea,
  CardActions,
  CardContent,
  CardMedia,
  Button,
  Typography
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'

import loading from '../../static/infinity.gif'

const useStyles = makeStyles({
  maxWidth: 512,
})

const Logo = ({
  modelName, 
  descrition, 
  isFetchingImage, 
  imgBase64,
  // fetchImage
}) => {
  const classes = useStyles()

  return (
    <Card className={classes.root}>
      <CardActionArea>
        <CardMedia
          component="img"
          alt="Logo"
          image={isFetchingImage ? loading : imgBase64}
          title="Logo"
        />
        <CardContent>
          <Typography gutterBottom variant="h5" component="h2">
            {modelName}
          </Typography>
          <Typography variant="body2" color="textSecondary" component="p">
            {descrition}
          </Typography>
        </CardContent>
      </CardActionArea>
      <CardActions>
        <Button size="large" color="primary">
          Download
        </Button>
      </CardActions>
    </Card>
  )
}

export default Logo
