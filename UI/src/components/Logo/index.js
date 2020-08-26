import { connect } from 'react-redux'

import Logo from "./Logo";
import { fetchImage } from '../ZGenerator/actions'

export default connect(
  ({ zGenerator: { isFetchingImage, imgBase64 } }) => ({ isFetchingImage, imgBase64 }),
  ({ fetchImage })
)(Logo)
