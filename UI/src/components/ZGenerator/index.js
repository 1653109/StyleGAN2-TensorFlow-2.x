import { connect } from 'react-redux'

import ZGenerator from "./ZGenerator";
import { fetchInfo, fetchImage } from './actions'

export default connect(
  ({ zGenerator: { isFetchingInfo, latentsDimensions } }) => ({ isFetchingInfo, latentsDimensions }),
  ({ fetchInfo, fetchImage })
)(ZGenerator)
