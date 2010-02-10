/**
 * 
 */
package sibtra.aco;

import sibtra.shm.ShmInterface;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;

/**
 * @author alberto
 *
 */
public class PanelACO extends PanelFlow {
	
	public PanelACO() {
		super();
		
		añadeAPanel(new LabelDatoFormato(sibtra.shm.ShmInterface.class, "getAcoLeftDist","%7d")
			,"Dist. Izda.");
		añadeAPanel(new LabelDatoFormato(sibtra.shm.ShmInterface.class, "getAcoRightDist","%7d")
		,"Dist. Dcha.");
		añadeAPanel(new LabelDatoFormato(sibtra.shm.ShmInterface.class, "getAcoHorizon","%7d")
		,"Horizonte");
		añadeAPanel(new LabelDatoFormato(sibtra.shm.ShmInterface.class, "getAcoRoadOrientation","%7d")
		,"Orientacion");
	}
	
	public void actualizaACO() {
		if(ShmInterface.getAcoAlive()!=0) { 
			actualizaDatos(this);
			ShmInterface.setAcoAlive(0);
		} else
			actualizaDatos(null);
	}

	/** En la actulización periodica  */
	protected void actualiza() {
		actualizaACO();
		super.actualiza();
	}

}
