import { connect } from 'react-redux'

import Modify from "./Modify";
import { changePsi, randomLatents, fetchImage } from '../ZGenerator/actions'

export default connect(
  ({ zGenerator: { psi } }) => ({ psi }),
  ({ changePsi, randomLatents, fetchImage })
)(Modify)
