package sibtra.util;

import java.awt.Dimension;
import java.awt.FlowLayout;

import javax.swing.JPanel;

/** Panel con FlowLayout que adapta su tamaño minimo para reflejar el número de 
 * filas que necesita
 * @author alberto
 */
public class PanelFlow extends JPanel {
	
	public PanelFlow() {
		super(new FlowLayout(FlowLayout.LEADING));
	}

	public Dimension getMinimumSize() {
	    Dimension sizePrefe = getPreferredSize();
	    Dimension sizeAct=getSize();
	    if(sizePrefe.width>sizeAct.width && sizeAct.width>0) {
	    	//hay partes que no se ven, aumentamos la altura de manera acorde
	    	sizePrefe.height=sizePrefe.height*(sizePrefe.width/sizeAct.width+1);
	    	sizePrefe.width=sizeAct.width;
	    }
	    return sizePrefe;
	}

}
