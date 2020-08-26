import { connect } from 'react-redux'

import Latent from "./Latent";
import { changeLatents, fetchImage } from '../ZGenerator/actions'

export default connect(
  ({ zGenerator: { latents } }) => ({ latents }),
  ({ changeLatents, fetchImage })
)(Latent)
